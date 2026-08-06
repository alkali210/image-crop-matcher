import json
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.imaging import read_image, resize_to_max, to_gray

CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    working_width: int
    working_height: int
    working_scale: float


@dataclass(frozen=True, slots=True)
class CoarseTemplateFeatures:
    pixels: np.ndarray
    offsets: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    image_indices: np.ndarray
    region_sizes: np.ndarray

    def __post_init__(self) -> None:
        for array in (
            self.pixels,
            self.offsets,
            self.widths,
            self.heights,
            self.image_indices,
            self.region_sizes,
        ):
            array.setflags(write=False)


class FeatureIndex:
    def __init__(
        self,
        image_ids: tuple[str, ...],
        by_image: dict[str, ImageFeatures],
        descriptors: np.ndarray,
        descriptor_image_indices: np.ndarray,
        coarse_templates: CoarseTemplateFeatures,
        loaded_from_cache: bool,
    ) -> None:
        self.image_ids = image_ids
        self.by_image = by_image
        self.descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        self.descriptor_image_indices = np.ascontiguousarray(
            descriptor_image_indices, dtype=np.int32
        )
        self.coarse_templates = CoarseTemplateFeatures(
            pixels=np.ascontiguousarray(coarse_templates.pixels, dtype=np.uint8),
            offsets=np.ascontiguousarray(coarse_templates.offsets, dtype=np.int64),
            widths=np.ascontiguousarray(coarse_templates.widths, dtype=np.int32),
            heights=np.ascontiguousarray(coarse_templates.heights, dtype=np.int32),
            image_indices=np.ascontiguousarray(coarse_templates.image_indices, dtype=np.int32),
            region_sizes=np.ascontiguousarray(coarse_templates.region_sizes, dtype=np.int32),
        )
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
            "gallery_root": catalog.root.as_posix(),
            "feature_settings": {
                "working_max_edge": settings.working_max_edge,
                "sift_features": settings.sift_features,
                "sift_contrast_threshold": settings.sift_contrast_threshold,
                "tile_sizes": list(settings.tile_sizes),
                "coarse_template_edge": settings.coarse_template_edge,
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
                    return cls._load(index_path, expected_image_ids, cache_identity, settings)
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
        coarse_levels: list[np.ndarray] = []
        coarse_image_indices: list[int] = []
        coarse_region_sizes: list[int] = []
        if settings.coarse_template_edge <= 0:
            raise ValueError("coarse_template_edge must be positive")

        for image_index, record in enumerate(catalog.records):
            safe_record = catalog.get(record.image_id)
            image, scale = resize_to_max(
                read_image(safe_record.path, settings.max_image_pixels),
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
            shorter_edge = min(width, height)
            for region_size in cls._region_sizes(shorter_edge, settings.tile_sizes):
                scale = settings.coarse_template_edge / region_size
                coarse_width = max(1, round(width * scale))
                coarse_height = max(1, round(height * scale))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                level = cv2.resize(
                    gray,
                    (coarse_width, coarse_height),
                    interpolation=interpolation,
                )
                coarse_levels.append(np.ascontiguousarray(level, dtype=np.uint8))
                coarse_image_indices.append(image_index)
                coarse_region_sizes.append(region_size)

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
            coarse_templates=cls._coarse_templates(
                coarse_levels, coarse_image_indices, coarse_region_sizes
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
            "coarse_pixels": self.coarse_templates.pixels,
            "coarse_offsets": self.coarse_templates.offsets,
            "coarse_widths": self.coarse_templates.widths,
            "coarse_heights": self.coarse_templates.heights,
            "coarse_image_indices": self.coarse_templates.image_indices,
            "coarse_region_sizes": self.coarse_templates.region_sizes,
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
        settings: Settings,
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
                    "coarse_pixels",
                    "coarse_offsets",
                    "coarse_widths",
                    "coarse_heights",
                    "coarse_image_indices",
                    "coarse_region_sizes",
                )
            }
        cls._validate_archive(arrays, expected_image_ids, expected_cache_identity, settings)

        image_ids = tuple(str(image_id) for image_id in arrays["image_ids"])
        points = arrays["points"]
        point_offsets = arrays["point_offsets"]
        descriptors = arrays["descriptors"]
        descriptor_offsets = arrays["descriptor_offsets"]
        working_widths = arrays["working_widths"]
        working_heights = arrays["working_heights"]
        working_scales = arrays["working_scales"]
        descriptor_image_indices = arrays["descriptor_image_indices"]
        coarse_templates = CoarseTemplateFeatures(
            pixels=np.ascontiguousarray(arrays["coarse_pixels"]),
            offsets=np.ascontiguousarray(arrays["coarse_offsets"]),
            widths=np.ascontiguousarray(arrays["coarse_widths"]),
            heights=np.ascontiguousarray(arrays["coarse_heights"]),
            image_indices=np.ascontiguousarray(arrays["coarse_image_indices"]),
            region_sizes=np.ascontiguousarray(arrays["coarse_region_sizes"]),
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
            coarse_templates=coarse_templates,
            loaded_from_cache=True,
        )

    @staticmethod
    def _validate_archive(
        arrays: dict[str, np.ndarray],
        expected_image_ids: tuple[str, ...],
        expected_cache_identity: str,
        settings: Settings,
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
            "coarse_pixels": np.dtype(np.uint8),
            "coarse_offsets": np.dtype(np.int64),
            "coarse_widths": np.dtype(np.int32),
            "coarse_heights": np.dtype(np.int32),
            "coarse_image_indices": np.dtype(np.int32),
            "coarse_region_sizes": np.dtype(np.int32),
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
        if len({str(image_id) for image_id in arrays["image_ids"]}) != image_count:
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

        coarse_names = (
            "coarse_pixels",
            "coarse_offsets",
            "coarse_widths",
            "coarse_heights",
            "coarse_image_indices",
            "coarse_region_sizes",
        )
        if any(arrays[name].ndim != 1 for name in coarse_names):
            raise ValueError("Invalid cached coarse template arrays")
        level_count = len(arrays["coarse_widths"])
        if any(
            len(arrays[name]) != level_count
            for name in (
                "coarse_heights",
                "coarse_image_indices",
                "coarse_region_sizes",
            )
        ):
            raise ValueError("Cached coarse template array lengths differ")
        coarse_offsets = arrays["coarse_offsets"]
        if (
            coarse_offsets.shape != (level_count + 1,)
            or coarse_offsets[0] != 0
            or np.any(np.diff(coarse_offsets) < 0)
            or coarse_offsets[-1] != len(arrays["coarse_pixels"])
        ):
            raise ValueError("Invalid cached coarse template offsets")
        coarse_widths = arrays["coarse_widths"].astype(np.int64)
        coarse_heights = arrays["coarse_heights"].astype(np.int64)
        coarse_region_sizes = arrays["coarse_region_sizes"]
        if (
            np.any(coarse_widths <= 0)
            or np.any(coarse_heights <= 0)
            or np.any(coarse_region_sizes <= 0)
        ):
            raise ValueError("Invalid cached coarse template dimensions")
        if not np.array_equal(np.diff(coarse_offsets), coarse_widths * coarse_heights):
            raise ValueError("Cached coarse template spans do not match dimensions")
        coarse_owners = arrays["coarse_image_indices"]
        if len(coarse_owners) and (coarse_owners.min() < 0 or coarse_owners.max() >= image_count):
            raise ValueError("Cached coarse template owner is out of range")
        if not np.array_equal(np.unique(coarse_owners), np.arange(image_count, dtype=np.int32)):
            raise ValueError("Cached coarse templates do not cover every image")

        expected_owners: list[int] = []
        expected_region_sizes: list[int] = []
        expected_widths: list[int] = []
        expected_heights: list[int] = []
        for image_index, (working_width, working_height) in enumerate(
            zip(arrays["working_widths"], arrays["working_heights"], strict=True)
        ):
            width = int(working_width)
            height = int(working_height)
            for region_size in FeatureIndex._region_sizes(min(width, height), settings.tile_sizes):
                scale = settings.coarse_template_edge / region_size
                expected_owners.append(image_index)
                expected_region_sizes.append(region_size)
                expected_widths.append(max(1, round(width * scale)))
                expected_heights.append(max(1, round(height * scale)))
        if not (
            np.array_equal(coarse_owners, np.asarray(expected_owners, dtype=np.int32))
            and np.array_equal(
                arrays["coarse_region_sizes"],
                np.asarray(expected_region_sizes, dtype=np.int32),
            )
            and np.array_equal(arrays["coarse_widths"], np.asarray(expected_widths, dtype=np.int32))
            and np.array_equal(
                arrays["coarse_heights"], np.asarray(expected_heights, dtype=np.int32)
            )
        ):
            raise ValueError("Cached coarse template metadata does not match settings")

    @staticmethod
    def _region_sizes(shorter_edge: int, tile_sizes: tuple[int, ...]) -> list[int]:
        configured_sizes = sorted({size for size in tile_sizes if size > 0})
        region_sizes = set(configured_sizes)
        region_sizes.update((left + right) // 2 for left, right in pairwise(configured_sizes))
        region_sizes = {size for size in region_sizes if size <= shorter_edge}
        if not region_sizes:
            region_sizes.add(shorter_edge)
        return sorted(region_sizes)

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
    def _coarse_templates(
        levels: list[np.ndarray], image_indices: list[int], region_sizes: list[int]
    ) -> CoarseTemplateFeatures:
        return CoarseTemplateFeatures(
            pixels=np.ascontiguousarray(
                np.concatenate([level.reshape(-1) for level in levels])
                if levels
                else np.empty(0, dtype=np.uint8),
                dtype=np.uint8,
            ),
            offsets=FeatureIndex._pixel_offsets(levels),
            widths=np.asarray([level.shape[1] for level in levels], dtype=np.int32),
            heights=np.asarray([level.shape[0] for level in levels], dtype=np.int32),
            image_indices=np.asarray(image_indices, dtype=np.int32),
            region_sizes=np.asarray(region_sizes, dtype=np.int32),
        )

    @staticmethod
    def _pixel_offsets(levels: list[np.ndarray]) -> np.ndarray:
        offsets = np.empty(len(levels) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([level.size for level in levels], out=offsets[1:])
        return offsets
