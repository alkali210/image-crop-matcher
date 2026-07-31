from collections.abc import Callable
from pathlib import Path
import shutil
from threading import Event, Thread

import numpy as np
import pytest

from crop_matcher.catalog import CatalogManifestEntry, ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex, TileFeatures
from crop_matcher.gallery_manager import (
    GalleryBundle,
    GalleryConflictError,
    GalleryManager,
    GalleryPathError,
)
from crop_matcher.gallery_state import GallerySelectionStore, gallery_cache_dir
from crop_matcher.matcher import ImageMatcher


def make_bundle(gallery_dir: Path, cache_dir: Path, settings: Settings) -> GalleryBundle:
    record = ImageRecord(
        image_id=gallery_dir.name,
        path=gallery_dir / "image.png",
        relative_path=Path("image.png"),
        parent_name=gallery_dir.name,
        filename="image.png",
        width=1,
        height=1,
    )
    catalog = ImageCatalog(
        gallery_dir,
        (record,),
        (CatalogManifestEntry("image.png", 1, 1),),
    )
    index = FeatureIndex(
        (record.image_id,),
        {},
        np.empty((0, 128), dtype=np.float32),
        np.empty(0, dtype=np.int32),
        TileFeatures(
            hashes=np.empty(0, dtype=np.uint64),
            image_indices=np.empty(0, dtype=np.int32),
            xs=np.empty(0, dtype=np.int32),
            ys=np.empty(0, dtype=np.int32),
            sizes=np.empty(0, dtype=np.int32),
        ),
        loaded_from_cache=False,
    )
    matcher = ImageMatcher(catalog, index, settings)
    return GalleryBundle(gallery_dir, cache_dir, catalog, index, matcher, 7)


@pytest.fixture
def first_gallery(tmp_path: Path) -> Path:
    gallery = tmp_path / "first"
    gallery.mkdir()
    return gallery


@pytest.fixture
def second_gallery(tmp_path: Path) -> Path:
    gallery = tmp_path / "second"
    gallery.mkdir()
    return gallery


@pytest.fixture
def broken_gallery(tmp_path: Path) -> Path:
    gallery = tmp_path / "broken"
    gallery.mkdir()
    return gallery


@pytest.fixture
def manager(tmp_path: Path, broken_gallery: Path) -> GalleryManager:
    settings = Settings(
        gallery_dir=tmp_path / "default",
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
    )

    def builder(gallery_dir: Path, cache_dir: Path) -> GalleryBundle:
        if gallery_dir == broken_gallery.resolve():
            raise RuntimeError(f"private failure at {gallery_dir}")
        return make_bundle(gallery_dir, cache_dir, settings)

    return GalleryManager(settings, GallerySelectionStore(settings.selection_file), builder=builder)


class BlockingBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started = Event()
        self.released = Event()
        self.calls = 0

    def __call__(self, gallery_dir: Path, cache_dir: Path) -> GalleryBundle:
        self.calls += 1
        if self.calls == 2:
            self.started.set()
            assert self.released.wait(timeout=5)
        return make_bundle(gallery_dir, cache_dir, self.settings)

    def start(self, target: Callable[[Path], None], path: Path) -> Thread:
        worker = Thread(target=target, args=(path,))
        worker.start()
        assert self.started.wait(timeout=5)
        return worker

    def release(self) -> None:
        self.released.set()


def test_failed_replacement_keeps_active_bundle(
    manager: GalleryManager, first_gallery: Path, broken_gallery: Path
) -> None:
    manager.initialize(first_gallery)
    old = manager.snapshot().active
    manager.reserve_switch(broken_gallery)
    manager.run_reserved_switch(broken_gallery)
    snapshot = manager.snapshot()
    assert snapshot.active is old
    assert snapshot.pending_gallery_dir is None
    assert snapshot.switch_error
    assert str(broken_gallery) not in snapshot.switch_error


def test_replacement_keeps_old_bundle_until_atomic_swap(
    tmp_path: Path, first_gallery: Path, second_gallery: Path
) -> None:
    settings = Settings(
        gallery_dir=first_gallery,
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
    )
    blocking_builder = BlockingBuilder(settings)
    manager = GalleryManager(
        settings,
        GallerySelectionStore(settings.selection_file),
        builder=blocking_builder,
    )
    manager.initialize(first_gallery)
    old = manager.snapshot().active
    manager.reserve_switch(second_gallery)
    worker = blocking_builder.start(manager.run_reserved_switch, second_gallery)
    assert manager.snapshot().active is old
    assert manager.snapshot().reindexing is True
    blocking_builder.release()
    worker.join(timeout=5)
    assert not worker.is_alive()
    snapshot = manager.snapshot()
    assert snapshot.active is not None
    assert snapshot.active.gallery_dir == second_gallery.resolve()
    assert GallerySelectionStore(settings.selection_file).load() == second_gallery.resolve()


def test_gallery_switch_reservation_validates_and_conflicts(
    manager: GalleryManager, first_gallery: Path, second_gallery: Path, tmp_path: Path
) -> None:
    manager.initialize(first_gallery)
    assert manager.reserve_switch(first_gallery) == "active"
    with pytest.raises(GalleryPathError):
        manager.reserve_switch(Path("relative"))
    with pytest.raises(GalleryPathError):
        manager.reserve_switch(tmp_path / "missing")
    assert manager.reserve_switch(second_gallery) == "accepted"
    with pytest.raises(GalleryConflictError):
        manager.reserve_switch(first_gallery)


def test_initialize_uses_persisted_gallery_and_namespaced_cache(
    tmp_path: Path, first_gallery: Path, second_gallery: Path
) -> None:
    settings = Settings(
        gallery_dir=first_gallery,
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
    )
    store = GallerySelectionStore(settings.selection_file)
    store.save(second_gallery)
    calls: list[tuple[Path, Path]] = []

    def builder(gallery_dir: Path, cache_dir: Path) -> GalleryBundle:
        calls.append((gallery_dir, cache_dir))
        return make_bundle(gallery_dir, cache_dir, settings)

    manager = GalleryManager(settings, store, builder=builder)
    manager.initialize()

    assert calls == [
        (second_gallery.resolve(), gallery_cache_dir(settings.cache_dir, second_gallery))
    ]
    assert manager.snapshot().state == "ready"


def test_failed_selection_save_rolls_back_built_gallery(
    tmp_path: Path, first_gallery: Path, second_gallery: Path
) -> None:
    settings = Settings(gallery_dir=first_gallery, cache_dir=tmp_path / "cache")

    class FailingStore(GallerySelectionStore):
        def save(self, gallery_dir: Path) -> None:
            raise OSError(f"cannot save {gallery_dir}")

    manager = GalleryManager(
        settings,
        FailingStore(tmp_path / "selection.json"),
        builder=lambda gallery_dir, cache_dir: make_bundle(gallery_dir, cache_dir, settings),
    )
    manager.initialize(first_gallery)
    old = manager.snapshot().active
    manager.reserve_switch(second_gallery)
    manager.run_reserved_switch(second_gallery)

    snapshot = manager.snapshot()
    assert snapshot.active is old
    assert snapshot.pending_gallery_dir is None
    assert snapshot.switch_error == "Failed to switch gallery"


def test_removed_reserved_gallery_clears_pending_switch(
    manager: GalleryManager, first_gallery: Path, second_gallery: Path
) -> None:
    manager.initialize(first_gallery)
    manager.reserve_switch(second_gallery)
    shutil.rmtree(second_gallery)

    manager.run_reserved_switch(second_gallery)

    snapshot = manager.snapshot()
    assert snapshot.pending_gallery_dir is None
    assert snapshot.reindexing is False
    assert snapshot.switch_error == "Failed to switch gallery"


def test_gallery_switch_conflicts_while_initial_bundle_is_building(
    tmp_path: Path, first_gallery: Path, second_gallery: Path
) -> None:
    settings = Settings(gallery_dir=first_gallery, cache_dir=tmp_path / "cache")
    started = Event()
    released = Event()

    def builder(gallery_dir: Path, cache_dir: Path) -> GalleryBundle:
        started.set()
        assert released.wait(timeout=5)
        return make_bundle(gallery_dir, cache_dir, settings)

    manager = GalleryManager(
        settings,
        GallerySelectionStore(tmp_path / "selection.json"),
        builder=builder,
    )
    worker = Thread(target=manager.initialize)
    worker.start()
    assert started.wait(timeout=5)

    try:
        with pytest.raises(GalleryConflictError):
            manager.reserve_switch(second_gallery)
    finally:
        released.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
