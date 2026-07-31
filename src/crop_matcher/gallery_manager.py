from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Literal

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.gallery_state import GallerySelectionStore, gallery_cache_dir
from crop_matcher.matcher import ImageMatcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GalleryBundle:
    gallery_dir: Path
    cache_dir: Path
    catalog: ImageCatalog
    feature_index: FeatureIndex
    matcher: ImageMatcher
    build_time_ms: int


@dataclass(frozen=True, slots=True)
class GallerySnapshot:
    state: Literal["building", "ready", "error"]
    active: GalleryBundle | None
    pending_gallery_dir: Path | None
    switch_error: str | None
    initial_error: str | None

    @property
    def reindexing(self) -> bool:
        return self.active is not None and self.pending_gallery_dir is not None


class GalleryPathError(ValueError):
    pass


class GalleryConflictError(RuntimeError):
    pass


GalleryBuilder = Callable[[Path, Path], GalleryBundle]


class GalleryManager:
    def __init__(
        self,
        settings: Settings,
        selection_store: GallerySelectionStore,
        builder: GalleryBuilder | None = None,
    ) -> None:
        self.settings = settings
        self.selection_store = selection_store
        self._builder = builder or self._build_bundle
        self._lock = Lock()
        self._snapshot = GallerySnapshot("building", None, None, None, None)

    def snapshot(self) -> GallerySnapshot:
        with self._lock:
            return self._snapshot

    def initialize(self, gallery_dir: Path | None = None) -> None:
        try:
            selected = gallery_dir
            if selected is None:
                selected = self.selection_store.load()
            if selected is None:
                selected = self.settings.gallery_dir
            resolved = selected.resolve(strict=False)
            cache_dir = gallery_cache_dir(self.settings.cache_dir, resolved)
            bundle = self._builder(resolved, cache_dir)
        except Exception as exc:
            logger.exception("Failed to initialize the image catalog and feature index")
            message = (
                "No supported images found"
                if isinstance(exc, (FileNotFoundError, NotADirectoryError, _EmptyGalleryError))
                else "Failed to initialize image index"
            )
            with self._lock:
                self._snapshot = GallerySnapshot("error", None, None, None, message)
            return

        with self._lock:
            self._snapshot = GallerySnapshot("ready", bundle, None, None, None)

    def reserve_switch(self, path: Path) -> Literal["active", "accepted"]:
        resolved = self._resolve_switch_path(path)
        with self._lock:
            snapshot = self._snapshot
            if snapshot.state == "building":
                raise GalleryConflictError("The initial gallery is still building")
            if snapshot.pending_gallery_dir is not None:
                raise GalleryConflictError("A gallery switch is already in progress")
            if snapshot.active is not None and snapshot.active.gallery_dir == resolved:
                return "active"
            self._snapshot = replace(
                snapshot,
                pending_gallery_dir=resolved,
                switch_error=None,
            )
        return "accepted"

    def run_reserved_switch(self, path: Path) -> None:
        with self._lock:
            reserved = self._snapshot.pending_gallery_dir
        try:
            candidate = path.resolve(strict=False)
            if reserved is None or reserved != candidate:
                raise GalleryConflictError("The gallery switch was not reserved")
            if not reserved.is_dir():
                raise GalleryPathError("Gallery path must be a directory")
            cache_dir = gallery_cache_dir(self.settings.cache_dir, reserved)
            bundle = self._builder(reserved, cache_dir)
            self.selection_store.save(reserved)
        except Exception:
            logger.exception("Failed to switch galleries")
            with self._lock:
                snapshot = self._snapshot
                if reserved is not None and snapshot.pending_gallery_dir is reserved:
                    self._snapshot = replace(
                        snapshot,
                        pending_gallery_dir=None,
                        switch_error="Failed to switch gallery",
                    )
            return

        with self._lock:
            if self._snapshot.pending_gallery_dir is not reserved:
                return
            self._snapshot = GallerySnapshot("ready", bundle, None, None, None)

    def _build_bundle(self, gallery_dir: Path, cache_dir: Path) -> GalleryBundle:
        started = time.perf_counter()
        catalog = ImageCatalog.load_or_scan(
            gallery_dir,
            self.settings.max_image_pixels,
            cache_dir / "catalog.json",
        )
        if not catalog.records:
            raise _EmptyGalleryError
        gallery_settings = replace(
            self.settings,
            gallery_dir=gallery_dir,
            cache_dir=cache_dir,
        )
        feature_index = FeatureIndex.load_or_build(catalog, gallery_settings)
        matcher = ImageMatcher(catalog, feature_index, gallery_settings)
        return GalleryBundle(
            gallery_dir,
            cache_dir,
            catalog,
            feature_index,
            matcher,
            round((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _resolve_switch_path(path: Path) -> Path:
        if not path.is_absolute():
            raise GalleryPathError("Gallery path must be absolute")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise GalleryPathError("Gallery path does not exist") from exc
        if not resolved.is_dir():
            raise GalleryPathError("Gallery path must be a directory")
        return resolved


class _EmptyGalleryError(RuntimeError):
    pass
