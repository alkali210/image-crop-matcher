from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.imaging import read_image, resize_to_max, to_gray


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    working_width: int
    working_height: int
    working_scale: float


@dataclass(frozen=True, slots=True)
class TileFeatures:
    hashes: np.ndarray
    image_indices: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    sizes: np.ndarray


class FeatureIndex:
    def __init__(
        self,
        image_ids: tuple[str, ...],
        by_image: dict[str, ImageFeatures],
        descriptors: np.ndarray,
        descriptor_image_indices: np.ndarray,
        tiles: TileFeatures,
        loaded_from_cache: bool,
    ) -> None:
        self.image_ids = image_ids
        self.by_image = by_image
        self.descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        self.descriptor_image_indices = np.ascontiguousarray(
            descriptor_image_indices, dtype=np.int32
        )
        self.tiles = tiles
        self.loaded_from_cache = loaded_from_cache
        self.global_matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 64},
        )
        if len(self.descriptors):
            self.global_matcher.add([self.descriptors])
            self.global_matcher.train()

    @classmethod
    def load_or_build(cls, catalog: ImageCatalog, settings: Settings) -> "FeatureIndex":
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = settings.cache_dir / "manifest.json"
        index_path = settings.cache_dir / "features.npz"
        manifest = [asdict(entry) for entry in catalog.manifest]
        if manifest_path.exists() and index_path.exists():
            if json.loads(manifest_path.read_text("utf-8")) == manifest:
                return cls._load(index_path)
        index = cls._build(catalog, settings)
        index._save(index_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
        return index

    @classmethod
    def _build(cls, catalog: ImageCatalog, settings: Settings) -> "FeatureIndex":
        sift = cv2.SIFT_create(
            nfeatures=settings.sift_features,
            contrastThreshold=settings.sift_contrast_threshold,
        )
        image_ids: list[str] = []
        by_image: dict[str, ImageFeatures] = {}
        descriptor_groups: list[np.ndarray] = []
        descriptor_image_groups: list[np.ndarray] = []

        for image_index, record in enumerate(catalog.records):
            image, scale = resize_to_max(
                read_image(record.path, settings.max_image_pixels),
                settings.working_max_edge,
            )
            keypoints, descriptors = sift.detectAndCompute(to_gray(image), None)
            if descriptors is None:
                points = np.empty((0, 2), dtype=np.float32)
                descriptors = np.empty((0, 128), dtype=np.float32)
            else:
                points = np.ascontiguousarray(
                    np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(
                        -1, 2
                    )
                )
                descriptors = np.ascontiguousarray(descriptors, dtype=np.float32).reshape(-1, 128)

            features = ImageFeatures(
                points=points,
                descriptors=descriptors,
                working_width=image.shape[1],
                working_height=image.shape[0],
                working_scale=scale,
            )
            image_ids.append(record.image_id)
            by_image[record.image_id] = features
            descriptor_groups.append(descriptors)
            descriptor_image_groups.append(np.full(len(descriptors), image_index, dtype=np.int32))

        all_descriptors = cls._concatenate_rows(descriptor_groups, 128, np.float32)
        if descriptor_image_groups:
            descriptor_image_indices = np.concatenate(descriptor_image_groups).astype(
                np.int32, copy=False
            )
        else:
            descriptor_image_indices = np.empty(0, dtype=np.int32)

        return cls(
            image_ids=tuple(image_ids),
            by_image=by_image,
            descriptors=all_descriptors,
            descriptor_image_indices=descriptor_image_indices,
            tiles=cls._empty_tiles(),
            loaded_from_cache=False,
        )

    def _save(self, index_path: Path) -> None:
        point_groups = [self.by_image[image_id].points for image_id in self.image_ids]
        descriptor_groups = [self.by_image[image_id].descriptors for image_id in self.image_ids]
        points = self._concatenate_rows(point_groups, 2, np.float32)
        descriptors = self._concatenate_rows(descriptor_groups, 128, np.float32)
        point_offsets = self._offsets(point_groups)
        descriptor_offsets = self._offsets(descriptor_groups)
        id_width = max((len(image_id) for image_id in self.image_ids), default=1)

        arrays = {
            "image_ids": np.asarray(self.image_ids, dtype=f"<U{id_width}"),
            "points": points,
            "point_offsets": point_offsets,
            "descriptors": descriptors,
            "descriptor_offsets": descriptor_offsets,
            "working_widths": np.asarray(
                [self.by_image[image_id].working_width for image_id in self.image_ids],
                dtype=np.int32,
            ),
            "working_heights": np.asarray(
                [self.by_image[image_id].working_height for image_id in self.image_ids],
                dtype=np.int32,
            ),
            "working_scales": np.asarray(
                [self.by_image[image_id].working_scale for image_id in self.image_ids],
                dtype=np.float64,
            ),
            "descriptor_image_indices": self.descriptor_image_indices,
            "tile_hashes": self.tiles.hashes,
            "tile_image_indices": self.tiles.image_indices,
            "tile_xs": self.tiles.xs,
            "tile_ys": self.tiles.ys,
            "tile_sizes": self.tiles.sizes,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{index_path.stem}.",
                suffix=".tmp.npz",
                dir=index_path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez(temporary_path, **arrays)
            temporary_path.replace(index_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def _load(cls, index_path: Path) -> "FeatureIndex":
        with np.load(index_path, allow_pickle=False) as cache:
            image_ids = tuple(str(image_id) for image_id in cache["image_ids"])
            points = np.asarray(cache["points"], dtype=np.float32)
            point_offsets = np.asarray(cache["point_offsets"], dtype=np.int64)
            descriptors = np.asarray(cache["descriptors"], dtype=np.float32)
            descriptor_offsets = np.asarray(cache["descriptor_offsets"], dtype=np.int64)
            working_widths = np.asarray(cache["working_widths"], dtype=np.int32)
            working_heights = np.asarray(cache["working_heights"], dtype=np.int32)
            working_scales = np.asarray(cache["working_scales"], dtype=np.float64)
            descriptor_image_indices = np.asarray(cache["descriptor_image_indices"], dtype=np.int32)
            tiles = TileFeatures(
                hashes=np.asarray(cache["tile_hashes"], dtype=np.uint64),
                image_indices=np.asarray(cache["tile_image_indices"], dtype=np.int32),
                xs=np.asarray(cache["tile_xs"], dtype=np.int32),
                ys=np.asarray(cache["tile_ys"], dtype=np.int32),
                sizes=np.asarray(cache["tile_sizes"], dtype=np.int32),
            )

        by_image: dict[str, ImageFeatures] = {}
        for image_index, image_id in enumerate(image_ids):
            point_start, point_end = point_offsets[image_index : image_index + 2]
            descriptor_start, descriptor_end = descriptor_offsets[image_index : image_index + 2]
            by_image[image_id] = ImageFeatures(
                points=np.ascontiguousarray(points[point_start:point_end], dtype=np.float32),
                descriptors=np.ascontiguousarray(
                    descriptors[descriptor_start:descriptor_end], dtype=np.float32
                ),
                working_width=int(working_widths[image_index]),
                working_height=int(working_heights[image_index]),
                working_scale=float(working_scales[image_index]),
            )

        return cls(
            image_ids=image_ids,
            by_image=by_image,
            descriptors=descriptors,
            descriptor_image_indices=descriptor_image_indices,
            tiles=tiles,
            loaded_from_cache=True,
        )

    @staticmethod
    def _concatenate_rows(groups: list[np.ndarray], width: int, dtype: np.dtype) -> np.ndarray:
        if not groups:
            return np.empty((0, width), dtype=dtype)
        return np.ascontiguousarray(np.concatenate(groups, axis=0), dtype=dtype).reshape(-1, width)

    @staticmethod
    def _offsets(groups: list[np.ndarray]) -> np.ndarray:
        offsets = np.empty(len(groups) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(group) for group in groups], out=offsets[1:])
        return offsets

    @staticmethod
    def _empty_tiles() -> TileFeatures:
        return TileFeatures(
            hashes=np.empty(0, dtype=np.uint64),
            image_indices=np.empty(0, dtype=np.int32),
            xs=np.empty(0, dtype=np.int32),
            ys=np.empty(0, dtype=np.int32),
            sizes=np.empty(0, dtype=np.int32),
        )
