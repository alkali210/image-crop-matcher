from hashlib import sha256
import os
from pathlib import Path

import pytest

from crop_matcher.gallery_state import GallerySelectionStore, gallery_cache_dir


def test_selection_store_round_trips_resolved_absolute_path(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    store = GallerySelectionStore(tmp_path / ".crop-matcher.json")
    assert store.load() is None
    store.save(gallery)
    assert store.load() == gallery.resolve()


def test_gallery_cache_namespaces_are_stable_and_distinct(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    normalized = os.path.normcase(str(first.resolve()))
    expected = tmp_path / ".cache" / "galleries" / sha256(normalized.encode("utf-8")).hexdigest()
    assert gallery_cache_dir(tmp_path / ".cache", first) == expected
    assert gallery_cache_dir(tmp_path / ".cache", first) == gallery_cache_dir(
        tmp_path / ".cache", first
    )
    assert gallery_cache_dir(tmp_path / ".cache", first) != gallery_cache_dir(
        tmp_path / ".cache", second
    )


def test_malformed_selection_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / ".crop-matcher.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="selection"):
        GallerySelectionStore(path).load()
