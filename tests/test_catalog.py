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


def test_ids_are_stable_and_unknown_ids_do_not_resolve(tmp_path: Path) -> None:
    write_image(tmp_path / "song" / "base.jpg")
    first = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    second = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    assert first.records[0].image_id == second.records[0].image_id
    with pytest.raises(KeyError):
        first.get("../../outside")
