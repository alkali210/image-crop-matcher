from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.imaging import perceptual_hash, read_image, resize_to_max, to_gray

CACHE_SCHEMA_VERSION = 2


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
        expected_image_ids = tuple(record.image_id for record in catalog.records)
        identity_source = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "feature_settings": {
                "working_max_edge": settings.working_max_edge,
                "sift_features": settings.sift_features,
                "sift_contrast_threshold": settings.sift_contrast_threshold,
                "tile_sizes": list(settings.tile_sizes),
            },
            "images": [asdict(entry) for entry in catalog.manifest],
        }
        cache_identity = sha256(
            json.dumps(
                identity_source,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        metadata = {**identity_source, "cache_identity": cache_identity}
        if manifest_path.exists() and index_path.exists():
            try:
                if json.loads(manifest_path.read_text("utf-8")) == metadata:
                    return cls._load(index_path, expected_image_ids, cache_identity)
            except Exception:
                # Cache data is disposable; source-image build errors remain outside this block.
                pass
        index = cls._build(catalog, settings)
        index._save(index_path, cache_identity)
        cls._save_manifest(manifest_path, metadata)
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
        tile_hashes: list[np.uint64] = []
        tile_image_indices: list[int] = []
        tile_xs: list[int] = []
        tile_ys: list[int] = []
        tile_sizes: list[int] = []

        for image_index, record in enumerate(catalog.records):
            image, scale = resize_to_max(
                read_image(record.path, settings.max_image_pixels),
                settings.working_max_edge,
            )
            gray = to_gray(image)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
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

            height, width = gray.shape[:2]
            for size in settings.tile_sizes:
                if size <= 0 or size > width or size > height:
                    continue
                stride = max(1, size // 2)
                xs = list(range(0, width - size + 1, stride))
                ys = list(range(0, height - size + 1, stride))
                if xs[-1] != width - size:
                    xs.append(width - size)
                if ys[-1] != height - size:
                    ys.append(height - size)
                for y in ys:
                    for x in xs:
                        tile_hashes.append(perceptual_hash(gray[y : y + size, x : x + size]))
                        tile_image_indices.append(image_index)
                        tile_xs.append(x)
                        tile_ys.append(y)
                        tile_sizes.append(size)

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
            tiles=TileFeatures(
                hashes=np.asarray(tile_hashes, dtype=np.uint64),
                image_indices=np.asarray(tile_image_indices, dtype=np.int32),
                xs=np.asarray(tile_xs, dtype=np.int32),
                ys=np.asarray(tile_ys, dtype=np.int32),
                sizes=np.asarray(tile_sizes, dtype=np.int32),
            ),
            loaded_from_cache=False,
        )

    def _save(self, index_path: Path, cache_identity: str) -> None:
        point_groups = [self.by_image[image_id].points for image_id in self.image_ids]
        descriptor_groups = [self.by_image[image_id].descriptors for image_id in self.image_ids]
        points = self._concatenate_rows(point_groups, 2, np.float32)
        descriptors = self._concatenate_rows(descriptor_groups, 128, np.float32)
        point_offsets = self._offsets(point_groups)
        descriptor_offsets = self._offsets(descriptor_groups)
        id_width = max((len(image_id) for image_id in self.image_ids), default=1)

        arrays = {
            "cache_identity": np.asarray(cache_identity, dtype=f"<U{len(cache_identity)}"),
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

    @staticmethod
    def _save_manifest(manifest_path: Path, metadata: dict[str, object]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f"{manifest_path.stem}.",
                suffix=".tmp.json",
                dir=manifest_path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(json.dumps(metadata, ensure_ascii=False))
            temporary_path.replace(manifest_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def _load(
        cls,
        index_path: Path,
        expected_image_ids: tuple[str, ...],
        expected_cache_identity: str,
    ) -> "FeatureIndex":
        with np.load(index_path, allow_pickle=False) as cache:
            arrays = {
                name: cache[name]
                for name in (
                    "cache_identity",
                    "image_ids",
                    "points",
                    "point_offsets",
                    "descriptors",
                    "descriptor_offsets",
                    "working_widths",
                    "working_heights",
                    "working_scales",
                    "descriptor_image_indices",
                    "tile_hashes",
                    "tile_image_indices",
                    "tile_xs",
                    "tile_ys",
                    "tile_sizes",
                )
            }
        cls._validate_archive(arrays, expected_image_ids, expected_cache_identity)

        image_ids = tuple(str(image_id) for image_id in arrays["image_ids"])
        points = arrays["points"]
        point_offsets = arrays["point_offsets"]
        descriptors = arrays["descriptors"]
        descriptor_offsets = arrays["descriptor_offsets"]
        working_widths = arrays["working_widths"]
        working_heights = arrays["working_heights"]
        working_scales = arrays["working_scales"]
        descriptor_image_indices = arrays["descriptor_image_indices"]
        tiles = TileFeatures(
            hashes=np.ascontiguousarray(arrays["tile_hashes"]),
            image_indices=np.ascontiguousarray(arrays["tile_image_indices"]),
            xs=np.ascontiguousarray(arrays["tile_xs"]),
            ys=np.ascontiguousarray(arrays["tile_ys"]),
            sizes=np.ascontiguousarray(arrays["tile_sizes"]),
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
    def _validate_archive(
        arrays: dict[str, np.ndarray],
        expected_image_ids: tuple[str, ...],
        expected_cache_identity: str,
    ) -> None:
        expected_dtypes = {
            "points": np.dtype(np.float32),
            "point_offsets": np.dtype(np.int64),
            "descriptors": np.dtype(np.float32),
            "descriptor_offsets": np.dtype(np.int64),
            "working_widths": np.dtype(np.int32),
            "working_heights": np.dtype(np.int32),
            "working_scales": np.dtype(np.float64),
            "descriptor_image_indices": np.dtype(np.int32),
            "tile_hashes": np.dtype(np.uint64),
            "tile_image_indices": np.dtype(np.int32),
            "tile_xs": np.dtype(np.int32),
            "tile_ys": np.dtype(np.int32),
            "tile_sizes": np.dtype(np.int32),
        }
        cache_identity = arrays["cache_identity"]
        if (
            cache_identity.shape != ()
            or cache_identity.dtype.kind != "U"
            or str(cache_identity.item()) != expected_cache_identity
        ):
            raise ValueError("Cached identity does not match metadata")
        if arrays["image_ids"].ndim != 1 or arrays["image_ids"].dtype.kind != "U":
            raise ValueError("Invalid cached image IDs")
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f"Invalid dtype for cached {name}")

        image_count = len(arrays["image_ids"])
        metadata_names = ("working_widths", "working_heights", "working_scales")
        if any(arrays[name].shape != (image_count,) for name in metadata_names):
            raise ValueError("Invalid cached per-image metadata")
        if len(set(str(image_id) for image_id in arrays["image_ids"])) != image_count:
            raise ValueError("Duplicate cached image IDs")
        if tuple(str(image_id) for image_id in arrays["image_ids"]) != expected_image_ids:
            raise ValueError("Cached image IDs do not match the catalog")
        if arrays["points"].ndim != 2 or arrays["points"].shape[1] != 2:
            raise ValueError("Invalid cached points")
        if arrays["descriptors"].ndim != 2 or arrays["descriptors"].shape[1] != 128:
            raise ValueError("Invalid cached descriptors")

        point_offsets = arrays["point_offsets"]
        descriptor_offsets = arrays["descriptor_offsets"]
        if point_offsets.shape != (image_count + 1,) or descriptor_offsets.shape != (
            image_count + 1,
        ):
            raise ValueError("Invalid cached offset count")
        if (
            point_offsets[0] != 0
            or descriptor_offsets[0] != 0
            or np.any(np.diff(point_offsets) < 0)
            or np.any(np.diff(descriptor_offsets) < 0)
        ):
            raise ValueError("Invalid cached offsets")
        if point_offsets[-1] != len(arrays["points"]) or descriptor_offsets[-1] != len(
            arrays["descriptors"]
        ):
            raise ValueError("Invalid cached terminal offsets")
        if not np.array_equal(np.diff(point_offsets), np.diff(descriptor_offsets)):
            raise ValueError("Cached point and descriptor rows differ")

        descriptor_owners = arrays["descriptor_image_indices"]
        if descriptor_owners.shape != (len(arrays["descriptors"]),):
            raise ValueError("Invalid cached descriptor owners")
        if len(descriptor_owners) and (
            descriptor_owners.min() < 0 or descriptor_owners.max() >= image_count
        ):
            raise ValueError("Cached descriptor owner is out of range")
        expected_owners = np.repeat(
            np.arange(image_count, dtype=np.int32), np.diff(descriptor_offsets)
        )
        if not np.array_equal(descriptor_owners, expected_owners):
            raise ValueError("Cached descriptor owners do not match offsets")

        tile_names = ("tile_hashes", "tile_image_indices", "tile_xs", "tile_ys", "tile_sizes")
        if any(arrays[name].ndim != 1 for name in tile_names):
            raise ValueError("Invalid cached tile arrays")
        tile_count = len(arrays["tile_hashes"])
        if any(len(arrays[name]) != tile_count for name in tile_names[1:]):
            raise ValueError("Cached tile array lengths differ")
        tile_owners = arrays["tile_image_indices"]
        if len(tile_owners) and (tile_owners.min() < 0 or tile_owners.max() >= image_count):
            raise ValueError("Cached tile owner is out of range")
        tile_xs = arrays["tile_xs"].astype(np.int64)
        tile_ys = arrays["tile_ys"].astype(np.int64)
        tile_sizes = arrays["tile_sizes"].astype(np.int64)
        if np.any(tile_xs < 0) or np.any(tile_ys < 0) or np.any(tile_sizes <= 0):
            raise ValueError("Invalid cached tile geometry")
        if len(tile_owners):
            tile_widths = arrays["working_widths"][tile_owners]
            tile_heights = arrays["working_heights"][tile_owners]
            if np.any(tile_xs + tile_sizes > tile_widths) or np.any(
                tile_ys + tile_sizes > tile_heights
            ):
                raise ValueError("Cached tile geometry is out of bounds")

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
