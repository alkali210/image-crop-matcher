from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Lock
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from crop_matcher import matcher as matcher_module
from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex, ImageFeatures, TileFeatures
from crop_matcher.matcher import CandidateScore, ImageMatcher


def make_art(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 70, (320, 320, 3), dtype=np.uint8)
    for index in range(18):
        center = tuple(int(value) for value in rng.integers(20, 300, 2))
        color = tuple(int(value) for value in rng.integers(100, 256, 3))
        cv2.circle(image, center, 5 + index, color, 2)
    cv2.putText(
        image,
        f"ART-{seed}",
        (70, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
    )
    return image


def write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path.write_bytes(payload.tobytes())


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(payload.tobytes())


class FakeGlobalMatcher:
    def __init__(self, matches: list[list[SimpleNamespace]]) -> None:
        self.matches = matches
        self.requested_k: int | None = None

    def knnMatch(self, _descriptors: np.ndarray, k: int) -> list[list[SimpleNamespace]]:
        self.requested_k = k
        return self.matches


class RecordingGlobalMatcher:
    def __init__(self, descriptors: np.ndarray) -> None:
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 64},
        )
        self.matcher.add([descriptors])
        self.matcher.train()
        self.requested_k: int | None = None

    def knnMatch(self, descriptors: np.ndarray, k: int) -> list[list[cv2.DMatch]]:
        self.requested_k = k
        return self.matcher.knnMatch(descriptors, k=k)


def neighbor(train_index: int, distance: float) -> SimpleNamespace:
    return SimpleNamespace(trainIdx=train_index, distance=distance)


def retrieval_matcher(
    image_ids: tuple[str, ...],
    owners: list[int],
    matches: list[list[SimpleNamespace]],
    candidate_count: int,
) -> tuple[ImageMatcher, FakeGlobalMatcher]:
    global_matcher = FakeGlobalMatcher(matches)
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.index = SimpleNamespace(
        descriptors=np.zeros((len(owners), 128), np.float32),
        descriptor_image_indices=np.asarray(owners, np.int32),
        image_ids=image_ids,
        global_matcher=global_matcher,
    )
    matcher.settings = Settings(candidate_count=candidate_count)
    matcher._flann_lock = Lock()
    return matcher, global_matcher


@pytest.mark.parametrize("grayscale", [False, True])
def test_matches_resized_crop_to_source(
    tmp_path: Path, grayscale: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    sources = [make_art(seed) for seed in range(3)]
    for seed, image in enumerate(sources):
        write_jpg(gallery / f"song-{seed}" / "base.jpg", image)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    index = FeatureIndex.load_or_build(catalog, settings)
    matcher = ImageMatcher(catalog, index, settings)
    query = cv2.resize(sources[1][90:210, 130:250], (90, 90))
    if grayscale:
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        query = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)

    def fail_fallback(_query_gray: np.ndarray) -> None:
        pytest.fail("textured primary fixture reached fallback")

    monkeypatch.setattr(matcher, "_fallback", fail_fallback)

    result = matcher.match(query)

    assert result.record.parent_name == "song-1"
    assert result.method == "sift"
    assert 0.0 <= result.similarity <= 100.0


def test_low_feature_crop_uses_fallback(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    first = np.zeros((320, 320, 3), np.uint8)
    second = np.zeros((320, 320, 3), np.uint8)
    cv2.rectangle(first, (40, 40), (280, 280), (80, 80, 80), -1)
    cv2.rectangle(first, (110, 110), (210, 210), (180, 180, 180), -1)
    cv2.rectangle(second, (40, 40), (280, 280), (80, 80, 80), -1)
    cv2.circle(second, (160, 160), 50, (180, 180, 180), -1)
    write_jpg(gallery / "square" / "base.jpg", first)
    write_jpg(gallery / "circle" / "base.jpg", second)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    matcher = ImageMatcher(catalog, FeatureIndex.load_or_build(catalog, settings), settings)
    query = cv2.resize(first[80:240, 80:240], (64, 64))

    result = matcher.match(query)

    assert result.record.parent_name == "square"
    assert result.method == "phash"
    assert result.similarity <= 89.9


def test_fallback_skips_invalid_and_duplicate_tile_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(b"fixture")
    second_path.write_bytes(b"fixture")
    records = (
        ImageRecord("first", first_path, Path("first.png"), "", "first.png", 1, 1),
        ImageRecord("second", second_path, Path("second.png"), "", "second.png", 1, 1),
    )
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.catalog = ImageCatalog(tmp_path, records, ())
    matcher.index = SimpleNamespace(
        image_ids=("first", "second"),
        tiles=TileFeatures(
            hashes=np.asarray([0, 1, 3, 7], np.uint64),
            image_indices=np.asarray([99, 0, 0, 1], np.int32),
            xs=np.zeros(4, np.int32),
            ys=np.zeros(4, np.int32),
            sizes=np.full(4, 64, np.int32),
        ),
    )
    matcher.settings = Settings(candidate_count=2)
    scored_indices: list[int] = []

    def template_score(image_index: int, _query_gray: np.ndarray) -> float:
        scored_indices.append(image_index)
        return (0.5, 0.9)[image_index]

    monkeypatch.setattr(matcher, "_template_score", template_score, raising=False)

    result = matcher._fallback(np.zeros((64, 64), np.uint8))

    assert scored_indices == [0, 1]
    assert result.record.image_id == "second"
    assert result.method == "phash"
    assert result.similarity <= 89.9


def test_template_score_uses_only_bounded_candidate_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    source = make_art(9)[:180, :240]
    write_png(gallery / "one" / "base.png", source)
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        working_max_edge=160,
        tile_sizes=(64, 96, 128),
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    record = catalog.records[0]
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.catalog = catalog
    matcher.settings = settings
    matcher.index = SimpleNamespace(
        image_ids=(record.image_id,),
        tiles=TileFeatures(
            hashes=np.zeros(3, np.uint64),
            image_indices=np.zeros(3, np.int32),
            xs=np.zeros(3, np.int32),
            ys=np.zeros(3, np.int32),
            sizes=np.asarray(settings.tile_sizes, np.int32),
        ),
    )
    calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def match_template(candidate: np.ndarray, query: np.ndarray, method: int) -> np.ndarray:
        assert method == cv2.TM_CCOEFF_NORMED
        calls.append((candidate.shape[:2], query.shape[:2]))
        response_shape = (
            candidate.shape[0] - query.shape[0] + 1,
            candidate.shape[1] - query.shape[1] + 1,
        )
        return np.zeros(response_shape, np.float32)

    monkeypatch.setattr(cv2, "matchTemplate", match_template)
    query = cv2.cvtColor(cv2.resize(source[40:120, 60:180], (120, 80)), cv2.COLOR_BGR2GRAY)

    score = matcher._template_score(0, query)

    assert 0.0 <= score <= 1.0
    assert len(calls) == 4
    assert all(query_shape == query.shape for _, query_shape in calls)
    assert all(
        query.shape[0] <= candidate_shape[0] <= 120 and query.shape[1] <= candidate_shape[1] <= 160
        for candidate_shape, _ in calls
    )


def test_retrieve_uses_five_neighbors_without_duplicate_owner_votes() -> None:
    matcher, global_matcher = retrieval_matcher(
        ("image-0", "image-1", "image-2"),
        [0, 0, 1, 2, 2],
        [
            [
                neighbor(0, 10.0),
                neighbor(1, 20.0),
                neighbor(2, 40.0),
                neighbor(3, 400.0),
                neighbor(4, 410.0),
            ],
            [
                neighbor(2, 14.0),
                neighbor(3, 20.0),
                neighbor(4, 21.0),
                neighbor(0, 22.0),
                neighbor(1, 23.0),
            ],
        ],
        candidate_count=2,
    )

    result = matcher._retrieve(np.zeros((2, 128), np.float32))

    assert global_matcher.requested_k == 5
    assert result == ["image-1", "image-0"]


def test_retrieve_ignores_unsafe_owners_and_deterministically_fills() -> None:
    matcher, _ = retrieval_matcher(
        tuple(f"image-{index}" for index in range(5)),
        [2, 99, 1, 3, 4],
        [
            [
                neighbor(0, 10.0),
                neighbor(1, 20.0),
                neighbor(99, 40.0),
                neighbor(2, 80.0),
                neighbor(3, 81.0),
            ]
        ],
        candidate_count=4,
    )

    result = matcher._retrieve(np.zeros((1, 128), np.float32))

    assert result == ["image-2", "image-0", "image-1", "image-3"]
    assert len(result) == len(set(result)) == 4


@pytest.mark.parametrize(
    ("descriptor_count", "expected_k", "expected"),
    [
        (1, None, ["image-0", "image-1", "image-2"]),
        (2, 2, ["image-2", "image-0", "image-1"]),
        (3, 3, ["image-2", "image-3", "image-0"]),
        (4, 4, ["image-2", "image-3", "image-4"]),
    ],
)
def test_retrieve_caps_knn_to_available_descriptor_rows(
    descriptor_count: int, expected_k: int | None, expected: list[str]
) -> None:
    descriptors = np.repeat(
        (10.0 * np.arange(descriptor_count, dtype=np.float32))[:, None],
        128,
        axis=1,
    )
    global_matcher = RecordingGlobalMatcher(descriptors)
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.index = SimpleNamespace(
        descriptors=descriptors,
        descriptor_image_indices=np.asarray([2, 3, 4, 1][:descriptor_count], np.int32),
        image_ids=tuple(f"image-{index}" for index in range(5)),
        global_matcher=global_matcher,
    )
    matcher.settings = Settings(candidate_count=3)
    matcher._flann_lock = Lock()

    result = matcher._retrieve(descriptors[:1])

    assert global_matcher.requested_k == expected_k
    assert result == expected
    assert len(result) == len(set(result)) == 3


def test_verify_accepts_large_scale_from_downscaled_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = cv2.resize(make_art(7), (64, 64), interpolation=cv2.INTER_AREA)
    source = np.zeros((1600, 1600, 3), np.uint8)
    source[32:1568, 32:1568] = np.repeat(np.repeat(query, 24, axis=0), 24, axis=1)
    gallery = tmp_path / "songs"
    write_png(gallery / "large" / "base.png", source)
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        working_max_edge=400,
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    record = catalog.records[0]
    query_points = np.asarray(
        [[0.0, 0.0], [252.0, 0.0], [252.0, 252.0], [0.0, 252.0]],
        np.float32,
    )
    feature_affine = np.asarray([[1.5, 0.0, 8.0], [0.0, 1.5, 8.0]], np.float64)
    candidate_points = cv2.transform(query_points[None, :, :], feature_affine)[0]
    descriptors = np.zeros((4, 128), np.float32)
    features = ImageFeatures(candidate_points, descriptors, 400, 400, 0.25)
    index = SimpleNamespace(by_image={record.image_id: features})
    matcher = ImageMatcher(catalog, index, settings)

    pairs = [
        [
            SimpleNamespace(queryIdx=index, trainIdx=index, distance=1.0),
            SimpleNamespace(queryIdx=index, trainIdx=(index + 1) % 4, distance=2.0),
        ]
        for index in range(4)
    ]
    local_matcher = SimpleNamespace(knnMatch=lambda *_args, **_kwargs: pairs)
    monkeypatch.setattr(cv2, "BFMatcher", lambda _norm: local_matcher)
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        lambda *_args, **_kwargs: (feature_affine, np.ones((4, 1), np.uint8)),
    )

    score = matcher._verify(
        record.image_id,
        cv2.cvtColor(query, cv2.COLOR_BGR2GRAY),
        query_points,
        descriptors,
        extraction_scale=4.0,
    )

    assert features.working_scale == pytest.approx(0.25)
    assert score is not None
    assert score.appearance > 0.95


def test_mapped_geometry_uses_per_axis_tolerance() -> None:
    valid = np.asarray([[0.0, -4.0], [999.0, -4.0], [999.0, 99.0], [0.0, 99.0]])
    outside_short_axis = np.asarray([[0.0, -6.0], [999.0, -6.0], [999.0, 99.0], [0.0, 99.0]])

    assert ImageMatcher._mapped_geometry_is_valid(valid, 1000, 100)
    assert not ImageMatcher._mapped_geometry_is_valid(outside_short_axis, 1000, 100)


def test_match_ranks_by_raw_score_and_applies_bounded_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best_record = ImageRecord("best", Path("best.png"), Path("best.png"), "", "best.png", 1, 1)
    second_record = ImageRecord(
        "second", Path("second.png"), Path("second.png"), "", "second.png", 1, 1
    )
    scores = {
        "best": CandidateScore(best_record, 0.5, 0.5, 0.45, 8, 0.8),
        "second": CandidateScore(second_record, 0.5, 0.5, 0.44, 8, 0.8),
    }
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher._sift = SimpleNamespace(
        detectAndCompute=lambda *_args: (
            [SimpleNamespace(pt=(float(index), float(index))) for index in range(4)],
            np.zeros((4, 128), np.float32),
        )
    )
    matcher._sift_lock = Lock()
    monkeypatch.setattr(matcher, "_feature_query", lambda query: (query, 1.0))
    monkeypatch.setattr(matcher, "_retrieve", lambda _descriptors: ["second", "best"])
    monkeypatch.setattr(matcher, "_verify", lambda image_id, *_args: scores[image_id])

    result = matcher.match(np.zeros((8, 8), np.uint8))

    assert result.record == best_record
    assert result.similarity == pytest.approx(45.5)
    assert 0.0 <= result.similarity <= 100.0


def test_simultaneous_matches_serialize_shared_sift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OverlapDetectingSift:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.lock = Lock()

        def detectAndCompute(self, *_args: object) -> tuple[list[object], None]:
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1
            return [], None

    matcher = ImageMatcher(ImageCatalog(Path(), (), ()), SimpleNamespace(), Settings())
    sift = OverlapDetectingSift()
    matcher._sift = sift
    expected = object()
    monkeypatch.setattr(matcher, "_fallback", lambda _query: expected)
    barrier = Barrier(2)

    def run_match() -> object:
        barrier.wait()
        return matcher.match(np.zeros((8, 8), np.uint8))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run_match(), range(2)))

    assert results == [expected, expected]
    assert sift.maximum_active == 1


def test_fallback_empty_index_uses_private_error() -> None:
    error_type = getattr(matcher_module, "_NoMatchEvidenceError", None)

    assert error_type is not None
    assert not hasattr(matcher_module, "NoMatchEvidenceError")
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.catalog = ImageCatalog(Path(), (), ())
    matcher.index = SimpleNamespace(
        image_ids=(),
        tiles=TileFeatures(
            hashes=np.empty(0, np.uint64),
            image_indices=np.empty(0, np.int32),
            xs=np.empty(0, np.int32),
            ys=np.empty(0, np.int32),
            sizes=np.empty(0, np.int32),
        ),
    )
    matcher.settings = Settings()

    with pytest.raises(error_type, match="Fallback index is empty"):
        matcher._fallback(np.zeros((8, 8), np.uint8))


def test_benchmark_queries_are_deterministic() -> None:
    from benchmarks.benchmark import crop_specs

    first = list(crop_specs(seed=20260730, image_count=3, samples_per_image=2))
    second = list(crop_specs(seed=20260730, image_count=3, samples_per_image=2))
    assert first == second
    assert {spec.output_size for spec in first}.issubset({64, 90, 128, 192})
    assert all(0.10 <= spec.crop_fraction <= 0.40 for spec in first)


def test_bounded_benchmark_reuses_isolated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks.benchmark import main

    gallery = tmp_path / "songs"
    write_png(gallery / "first" / "base.png", make_art(1))
    write_png(gallery / "second" / "base.png", make_art(2))
    production_cache = tmp_path / ".cache"
    production_cache.mkdir()
    sentinel = production_cache / "full-gallery-sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    arguments = [
        "--gallery",
        str(gallery),
        "--samples-per-image",
        "1",
        "--max-images",
        "1",
        "--failure-dir",
        str(tmp_path / "failures"),
    ]

    assert main(arguments) == 0
    capsys.readouterr()
    namespaces = list((production_cache / "benchmarks").iterdir())
    assert len(namespaces) == 1
    manifest = namespaces[0] / "manifest.json"
    features = namespaces[0] / "features.npz"
    assert manifest.is_file()
    assert features.is_file()
    timestamps = (manifest.stat().st_mtime_ns, features.stat().st_mtime_ns)

    assert main(arguments) == 0
    capsys.readouterr()
    assert list((production_cache / "benchmarks").iterdir()) == namespaces
    assert (manifest.stat().st_mtime_ns, features.stat().st_mtime_ns) == timestamps
    assert sentinel.read_text("utf-8") == "keep"
    assert not (production_cache / "manifest.json").exists()
    assert not (production_cache / "features.npz").exists()


def test_benchmark_accuracy_exit_status_uses_95_percent_boundary() -> None:
    from benchmarks.benchmark import accuracy_exit_status

    assert accuracy_exit_status(1.0) == 0
    assert accuracy_exit_status(0.95) == 0
    assert accuracy_exit_status(0.9499) == 1


def test_benchmark_failure_json_has_exact_metadata(tmp_path: Path) -> None:
    from benchmarks.benchmark import CropSpec, write_failure

    spec = CropSpec(1, 2, 0.25, 0.5, 0.75, 90, True)
    path = write_failure(
        tmp_path,
        7,
        source_id="source",
        predicted_id="predicted",
        x=11,
        y=13,
        side=101,
        spec=spec,
        similarity=72.5,
        latency_ms=4.5678,
    )

    assert path.name == "failure-000007.json"
    assert json.loads(path.read_text("utf-8")) == {
        "source_id": "source",
        "predicted_id": "predicted",
        "crop_x": 11,
        "crop_y": 13,
        "crop_side": 101,
        "output_size": 90,
        "grayscale": True,
        "similarity": 72.5,
        "latency_ms": 4.568,
    }


def test_benchmark_rejects_unowned_failure_run_without_deleting_files(
    tmp_path: Path,
) -> None:
    from benchmarks.benchmark import failure_run_dir, prepare_failure_run_dir

    failure_root = tmp_path / "caller-owned"
    failure_root.mkdir()
    caller_failure = failure_root / "failure-caller.json"
    caller_failure.write_text("keep", encoding="utf-8")
    gallery = tmp_path / "songs"
    run_dir = failure_run_dir(failure_root, gallery, 1, 2, 20260730)
    run_dir.mkdir()
    run_failure = run_dir / "failure-000001.json"
    run_failure.write_text("also keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not benchmark-owned"):
        prepare_failure_run_dir(failure_root, gallery, 1, 2, 20260730)

    assert caller_failure.read_text("utf-8") == "keep"
    assert run_failure.read_text("utf-8") == "also keep"


def test_benchmark_cleans_owned_failure_run_before_catalog_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.benchmark import main, prepare_failure_run_dir

    gallery = tmp_path / "songs"
    failure_root = tmp_path / "failures"
    run_dir = prepare_failure_run_dir(failure_root, gallery, 1, 1, 20260730)
    stale = run_dir / "failure-999999.json"
    stale.write_text("stale", encoding="utf-8")

    def fail_scan(_cls: type[ImageCatalog], _root: Path, _max_pixels: int) -> None:
        assert not stale.exists()
        raise RuntimeError("catalog stopped")

    monkeypatch.setattr(ImageCatalog, "scan", classmethod(fail_scan))

    with pytest.raises(RuntimeError, match="catalog stopped"):
        main(
            [
                "--gallery",
                str(gallery),
                "--samples-per-image",
                "1",
                "--max-images",
                "1",
                "--failure-dir",
                str(failure_root),
            ]
        )


def test_benchmark_seeds_opencv_before_catalog_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import benchmark

    calls: list[int] = []
    monkeypatch.setattr(benchmark.cv2, "setRNGSeed", calls.append)

    def fail_scan(_cls: type[ImageCatalog], _root: Path, _max_pixels: int) -> None:
        assert calls == [20260730]
        raise RuntimeError("catalog stopped")

    monkeypatch.setattr(ImageCatalog, "scan", classmethod(fail_scan))

    with pytest.raises(RuntimeError, match="catalog stopped"):
        benchmark.main(
            [
                "--gallery",
                str(tmp_path / "songs"),
                "--samples-per-image",
                "1",
                "--failure-dir",
                str(tmp_path / "failures"),
            ]
        )


def test_benchmark_owned_failure_lifecycle_and_status_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks.benchmark import failure_run_dir, main
    from crop_matcher.matcher import MatchResult

    gallery = tmp_path / "songs"
    write_png(gallery / "first" / "base.png", make_art(1))
    write_png(gallery / "second" / "base.png", make_art(2))
    failure_root = tmp_path / "caller-owned"
    failure_root.mkdir()
    caller_failure = failure_root / "failure-caller.json"
    caller_failure.write_text("keep", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def always_first(matcher: ImageMatcher, _query: np.ndarray) -> MatchResult:
        return MatchResult(matcher.catalog.records[0], 55.0, "sift", 4, 1.0, 1.0)

    monkeypatch.setattr(ImageMatcher, "match", always_first)
    arguments = [
        "--gallery",
        str(gallery),
        "--samples-per-image",
        "1",
        "--max-images",
        "2",
        "--failure-dir",
        str(failure_root),
    ]

    assert main(arguments) == 1
    capsys.readouterr()
    run_dir = failure_run_dir(failure_root, gallery, 1, 2, 20260730)
    assert (run_dir / ".benchmark-owned").is_file()
    assert caller_failure.read_text("utf-8") == "keep"
    failures = list(run_dir.glob("failure-*.json"))
    assert [path.name for path in failures] == ["failure-000002.json"]
    metadata = json.loads(failures[0].read_text("utf-8"))
    assert metadata["source_id"] != metadata["predicted_id"]

    stale = run_dir / "failure-999999.json"
    stale.write_text("stale", encoding="utf-8")
    owned_non_failure = run_dir / "keep.txt"
    owned_non_failure.write_text("keep", encoding="utf-8")

    assert main(arguments) == 1
    capsys.readouterr()
    assert not stale.exists()
    assert owned_non_failure.read_text("utf-8") == "keep"
    assert caller_failure.read_text("utf-8") == "keep"
    assert [path.name for path in run_dir.glob("failure-*.json")] == ["failure-000002.json"]


def test_benchmark_cli_returns_zero_at_accuracy_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks.benchmark import main

    gallery = tmp_path / "songs"
    write_png(gallery / "only" / "base.png", make_art(7))
    monkeypatch.chdir(tmp_path)

    exit_status = main(
        [
            "--gallery",
            str(gallery),
            "--samples-per-image",
            "1",
        ]
    )

    output = capsys.readouterr().out.splitlines()
    assert exit_status == 0
    assert output[:2] == ["images=1 queries=1", "top1=1/1 accuracy=100.00%"]
    assert output[2] in {"method_sift=1 method_phash=0", "method_sift=0 method_phash=1"}
    assert output[3].startswith("latency_ms_p50=")
    assert " latency_ms_p95=" in output[3]
    assert (tmp_path / "benchmark-failures").is_dir()
