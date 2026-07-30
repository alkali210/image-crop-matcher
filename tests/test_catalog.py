import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from crop_matcher.catalog import ImageCatalog


def write_image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", np.full((20, 20, 3), value, np.uint8))
    assert ok
    path.write_bytes(payload.tobytes())


def create_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")


def test_scan_is_sorted_filters_thumbnails_and_skips_corrupt_files(tmp_path: Path) -> None:
    write_image(tmp_path / "z-song" / "cover.PNG", 20)
    write_image(tmp_path / "a-song" / "base.jpg", 30)
    write_image(tmp_path / "a-song" / "base_256.jpg", 40)
    (tmp_path / "bad.jpg").write_bytes(b"broken")
    (tmp_path / "audio.wav").write_bytes(b"audio")

    catalog = ImageCatalog.scan(tmp_path, max_pixels=10_000)

    assert [record.relative_path.as_posix() for record in catalog.records] == [
        "a-song/base.jpg",
        "z-song/cover.PNG",
    ]
    assert catalog.records[0].parent_name == "a-song"
    assert catalog.get(catalog.records[0].image_id) == catalog.records[0]


def test_scan_skips_opencv_decode_failure_and_keeps_valid_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "a-broken.png"
    valid = tmp_path / "b-valid.png"
    write_image(broken, 20)
    write_image(valid, 30)
    broken_payload = broken.read_bytes()
    original_imdecode = cv2.imdecode

    def selective_decode(data: np.ndarray, flags: int) -> np.ndarray:
        if data.tobytes() == broken_payload:
            raise cv2.error("OpenCV decode failed")
        return original_imdecode(data, flags)

    monkeypatch.setattr(cv2, "imdecode", selective_decode)

    catalog = ImageCatalog.scan(tmp_path, max_pixels=10_000)

    assert [record.relative_path.as_posix() for record in catalog.records] == ["b-valid.png"]


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("corrupt", "not a supported image"),
        ("oversized", "exceeds the pixel limit"),
        ("outside", "outside.png"),
        ("filesystem", "stat failed"),
    ],
)
def test_scan_warns_for_skipped_source_and_keeps_valid_peers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    reason: str,
) -> None:
    gallery = tmp_path / "gallery"
    bad = gallery / f"bad-{failure}.png"
    valid = gallery / "valid.png"
    write_image(valid, 30)
    max_pixels = 10_000

    if failure == "corrupt":
        bad.write_bytes(b"broken")
    elif failure == "oversized":
        write_image(bad, 20)
        ok, payload = cv2.imencode(".png", np.full((5, 5, 3), 30, np.uint8))
        assert ok
        valid.write_bytes(payload.tobytes())
        max_pixels = 100
    elif failure == "outside":
        outside = tmp_path / "outside.png"
        write_image(outside, 20)
        create_symlink_or_skip(bad, outside)
    else:
        write_image(bad, 20)
        original_stat = Path.stat

        def selective_stat(path: Path, *args: object, **kwargs: object) -> object:
            if path.name == bad.name:
                raise OSError("stat failed")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", selective_stat)

    with caplog.at_level(logging.WARNING, logger="crop_matcher.catalog"):
        catalog = ImageCatalog.scan(gallery, max_pixels=max_pixels)

    assert [record.relative_path.as_posix() for record in catalog.records] == ["valid.png"]
    warning = next(record for record in caplog.records if bad.name in record.getMessage())
    assert warning.levelno == logging.WARNING
    assert reason in warning.getMessage()


def test_scan_breaks_casefold_sort_ties_with_original_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sharp_s_path = tmp_path / "ß.jpg"
    double_s_path = tmp_path / "ss.jpg"
    write_image(sharp_s_path, 20)
    write_image(double_s_path, 30)
    monkeypatch.setattr(Path, "rglob", lambda _root, _pattern: iter([sharp_s_path, double_s_path]))

    catalog = ImageCatalog.scan(tmp_path, max_pixels=10_000)

    expected = ["ss.jpg", "ß.jpg"]
    assert [record.relative_path.as_posix() for record in catalog.records] == expected
    assert [entry.relative_path for entry in catalog.manifest] == expected


def test_ids_are_stable_and_unknown_ids_do_not_resolve(tmp_path: Path) -> None:
    write_image(tmp_path / "song" / "base.jpg")
    first = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    second = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    assert first.records[0].image_id == second.records[0].image_id
    with pytest.raises(KeyError):
        first.get("../../outside")


def test_get_rejects_record_whose_resolved_path_is_outside_root(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    outside = tmp_path / "outside.png"
    write_image(outside)
    record = ImageCatalog.scan(tmp_path, max_pixels=10_000).records[0]
    catalog = ImageCatalog(gallery, (record,), ())

    with pytest.raises(KeyError):
        catalog.get(record.image_id)


def test_scan_skips_external_file_symlink(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    outside = tmp_path / "outside.png"
    write_image(outside)
    create_symlink_or_skip(gallery / "linked.png", outside)

    catalog = ImageCatalog.scan(gallery, max_pixels=10_000)

    assert catalog.records == ()


def test_get_rejects_file_symlink_replaced_after_scan(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    source = gallery / "source.png"
    outside = tmp_path / "outside.png"
    write_image(source, 20)
    write_image(outside, 30)
    catalog = ImageCatalog.scan(gallery, max_pixels=10_000)
    image_id = catalog.records[0].image_id
    source.unlink()
    create_symlink_or_skip(source, outside)

    with pytest.raises(KeyError):
        catalog.get(image_id)
