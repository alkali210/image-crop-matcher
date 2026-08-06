import json
import re
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from benchmarks import example_regression
from benchmarks.benchmark import DEFAULT_SEED
from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.matcher import ImageMatcher, MatchResult


def make_art(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 70, (240, 240, 3), dtype=np.uint8)
    cv2.circle(image, (120, 120), 70, (220, 180, 140), 5)
    cv2.putText(
        image,
        f"ART-{seed}",
        (45, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, payload = cv2.imencode(suffix, image)
    assert ok
    path.write_bytes(payload.tobytes())


def regression_fixture(tmp_path: Path) -> tuple[Path, Path]:
    gallery = tmp_path / "songs"
    examples = tmp_path / "examples"
    source = make_art(7)
    write_image(gallery / "only" / "base.jpg", source)
    write_image(examples / "query.png", source[40:200, 40:200])
    (examples / "expected.json").write_text(
        json.dumps({"query.png": "only/base.jpg"}),
        encoding="utf-8",
    )
    return gallery, examples


@pytest.mark.parametrize(
    ("manifest", "query_payload", "message"),
    [
        ([], None, "JSON object"),
        ({}, None, "at least one"),
        ({"": "only/base.jpg"}, None, "nonempty string"),
        ({"query.png": ""}, None, "nonempty string"),
        ({"missing.png": "only/base.jpg"}, None, "does not exist"),
        ({"query.png": "missing/base.jpg"}, b"valid", "not in the gallery"),
        ({"query.png": "only/base.jpg"}, b"broken", "not a supported image"),
    ],
)
def test_manifest_validation_rejects_invalid_setup(
    tmp_path: Path,
    manifest: object,
    query_payload: bytes | None,
    message: str,
) -> None:
    gallery = tmp_path / "songs"
    examples = tmp_path / "examples"
    write_image(gallery / "only" / "base.jpg", make_art(1))
    examples.mkdir()
    (examples / "expected.json").write_text(json.dumps(manifest), encoding="utf-8")
    if query_payload == b"valid":
        write_image(examples / "query.png", make_art(2))
    elif query_payload is not None:
        (examples / "query.png").write_bytes(query_payload)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    with pytest.raises((TypeError, ValueError, FileNotFoundError), match=message):
        example_regression.load_cases(examples, catalog, settings.max_image_pixels)


def test_manifest_validation_rejects_malformed_json(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "expected.json").write_text("not-json", encoding="utf-8")
    catalog = ImageCatalog(tmp_path / "songs", (), ())

    with pytest.raises(ValueError, match="expected.json"):
        example_regression.load_cases(examples, catalog, 10_000)


def test_manifest_validation_rejects_duplicate_query_keys(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "expected.json").write_text(
        '{"query.png": "one/base.jpg", "query.png": "two/base.jpg"}',
        encoding="utf-8",
    )
    catalog = ImageCatalog(tmp_path / "songs", (), ())

    with pytest.raises(ValueError, match=r"Duplicate.*query\.png"):
        example_regression.load_cases(examples, catalog, 10_000)


def test_parser_defaults() -> None:
    arguments = example_regression._parser().parse_args([])

    assert arguments.gallery == Path("songs")
    assert arguments.examples == Path("examples")
    assert arguments.max_p95_ms == 1000.0


def test_direct_script_entrypoint_is_importable() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmarks/example_regression.py", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run real-example crop matching regression cases" in completed.stdout


def test_run_seeds_opencv_before_catalog_scan_and_index_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery, examples = regression_fixture(tmp_path)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    events: list[object] = []
    monkeypatch.setattr(
        example_regression,
        "seed_opencv",
        lambda seed: events.append(("seed", seed)),
        raising=False,
    )
    monkeypatch.setattr(
        ImageCatalog,
        "scan",
        classmethod(lambda _cls, *_args: (events.append("scan"), catalog)[1]),
    )

    def stop_after_build_order(*_args: object) -> None:
        events.append("build")
        raise RuntimeError("stop after order check")

    monkeypatch.setattr(FeatureIndex, "load_or_build", stop_after_build_order)

    with pytest.raises(RuntimeError, match="order check"):
        example_regression.run_regression(gallery, examples, 1000.0)

    assert events == [("seed", DEFAULT_SEED), "scan", "build"]


def test_tiny_real_gallery_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gallery, examples = regression_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    status = example_regression.main(["--gallery", str(gallery), "--examples", str(examples)])

    output = capsys.readouterr().out.splitlines()
    assert status == 0
    assert len(output) == 2
    assert re.fullmatch(
        r"query=query\.png expected=only/base\.jpg predicted=only/base\.jpg "
        r"method=(sift|template) similarity=\d+\.\d latency_ms=\d+\.\d{3} correct=true",
        output[0],
    )
    assert re.fullmatch(r"top1=1/1 accuracy=100\.00% latency_ms_p95=\d+\.\d{3}", output[1])


def test_incorrect_result_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gallery, examples = regression_fixture(tmp_path)
    wrong = gallery / "wrong" / "base.jpg"
    write_image(wrong, make_art(8))
    original_match = ImageMatcher.match

    def wrong_timed_match(matcher: ImageMatcher, query: np.ndarray) -> MatchResult:
        result = original_match(matcher, query)
        if wrong_timed_match.calls == 0:
            wrong_timed_match.calls += 1
            return result
        wrong_record = next(
            record
            for record in matcher.catalog.records
            if record.relative_path == Path("wrong/base.jpg")
        )
        return MatchResult(wrong_record, 12.3, "template", 0, 0.0, 0.1)

    wrong_timed_match.calls = 0
    monkeypatch.setattr(ImageMatcher, "match", wrong_timed_match)
    monkeypatch.chdir(tmp_path)

    status = example_regression.main(["--gallery", str(gallery), "--examples", str(examples)])

    output = capsys.readouterr().out.splitlines()
    assert status == 1
    assert "predicted=wrong/base.jpg" in output[0]
    assert output[0].endswith("correct=false")
    assert output[1].startswith("top1=0/1 accuracy=0.00%")


def test_latency_threshold_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gallery, examples = regression_fixture(tmp_path)
    ticks = iter((10_000_000, 12_000_000))
    monkeypatch.setattr(example_regression, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.chdir(tmp_path)

    status = example_regression.main(
        [
            "--gallery",
            str(gallery),
            "--examples",
            str(examples),
            "--max-p95-ms",
            "1",
        ]
    )

    output = capsys.readouterr().out.splitlines()
    assert status == 1
    assert "latency_ms=2.000" in output[0]
    assert output[1] == "top1=1/1 accuracy=100.00% latency_ms_p95=2.000"
