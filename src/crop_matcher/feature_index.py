from collections import defaultdict
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.imaging import read_image, resize_to_max, to_gray

CACHE_SCHEMA_VERSION = 9
FEATURE_INDEX_DIR_PREFIX = "feature-index-"
REPRESENTATIVES_PER_IMAGE = 128
REPRESENTATIVE_GRID_SIZE = 8

_ARRAY_NAMES = (
    "cache_identity",
    "image_ids",
    "points",
    "point_offsets",
    "descriptors",
    "descriptor_offsets",
    "working_widths",
    "working_heights",
    "working_scales",
    "coarse_pixels",
    "coarse_offsets",
    "coarse_widths",
    "coarse_heights",
    "coarse_image_indices",
    "coarse_region_sizes",
    "representative_descriptors",
    "representative_image_indices",
)
_MMAP_ARRAYS = frozenset(
    {
        "points",
        "descriptors",
        "coarse_pixels",
        "representative_descriptors",
    }
)


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
        points: np.ndarray,
        point_offsets: np.ndarray,
        descriptors: np.ndarray,
        descriptor_offsets: np.ndarray,
        working_widths: np.ndarray,
        working_heights: np.ndarray,
        working_scales: np.ndarray,
        coarse_templates: CoarseTemplateFeatures,
        representative_descriptors: np.ndarray,
        representative_image_indices: np.ndarray,
        loaded_from_cache: bool,
    ) -> None:
        self.image_ids = image_ids
        self.points = self._typed_array(points, np.float32)
        self.point_offsets = self._typed_array(point_offsets, np.int64)
        self.descriptors = self._typed_array(descriptors, np.uint8)
        self.descriptor_offsets = self._typed_array(descriptor_offsets, np.int64)
        self.working_widths = self._typed_array(working_widths, np.int32)
        self.working_heights = self._typed_array(working_heights, np.int32)
        self.working_scales = self._typed_array(working_scales, np.float64)
        self.coarse_templates = CoarseTemplateFeatures(
            pixels=self._typed_array(coarse_templates.pixels, np.uint8),
            offsets=self._typed_array(coarse_templates.offsets, np.int64),
            widths=self._typed_array(coarse_templates.widths, np.int32),
            heights=self._typed_array(coarse_templates.heights, np.int32),
            image_indices=self._typed_array(coarse_templates.image_indices, np.int32),
            region_sizes=self._typed_array(coarse_templates.region_sizes, np.int32),
        )
        self.representative_descriptors = self._typed_array(
            representative_descriptors,
            np.float32,
        )
        self.representative_image_indices = self._typed_array(
            representative_image_indices,
            np.int32,
        )
        self.loaded_from_cache = loaded_from_cache

        self.by_image: dict[str, ImageFeatures] = {}
        for image_index, image_id in enumerate(image_ids):
            point_start, point_end = self.point_offsets[image_index : image_index + 2]
            descriptor_start, descriptor_end = self.descriptor_offsets[
                image_index : image_index + 2
            ]
            self.by_image[image_id] = ImageFeatures(
                points=self.points[point_start:point_end],
                descriptors=self.descriptors[descriptor_start:descriptor_end],
                working_width=int(self.working_widths[image_index]),
                working_height=int(self.working_heights[image_index]),
                working_scale=float(self.working_scales[image_index]),
            )

        self._global_matcher: cv2.FlannBasedMatcher | None = None
        if len(self.representative_descriptors):
            self._global_matcher = cv2.FlannBasedMatcher(
                {"algorithm": 1, "trees": 5},
                {"checks": 64},
            )
            self._global_matcher.add([self.representative_descriptors])
            self._global_matcher.train()

    @staticmethod
    def _typed_array(array: np.ndarray, dtype: Any) -> np.ndarray:
        target_dtype = np.dtype(dtype)
        if array.dtype != target_dtype or not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, dtype=target_dtype)
        array.setflags(write=False)
        return array

    @classmethod
    def load_or_build(cls, catalog: ImageCatalog, settings: Settings) -> "FeatureIndex":
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = settings.cache_dir / "manifest.json"
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
                "representatives_per_image": REPRESENTATIVES_PER_IMAGE,
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
        if manifest_path.exists():
            try:
                existing_metadata = cast(
                    dict[str, object],
                    json.loads(manifest_path.read_text("utf-8")),
                )
                index_name = existing_metadata.pop("feature_index_dir")
                if (
                    existing_metadata == metadata
                    and isinstance(index_name, str)
                    and Path(index_name).name == index_name
                    and index_name.startswith(FEATURE_INDEX_DIR_PREFIX)
                ):
                    index_dir = settings.cache_dir / index_name
                    index = cls._load(
                        index_dir,
                        expected_image_ids,
                        cache_identity,
                        settings,
                        loaded_from_cache=True,
                    )
                    cls._cleanup_index_dirs(settings.cache_dir, index_name)
                    return index
            except Exception:
                # Cache data is disposable; source-image build errors remain outside this block.
                pass

        index_name = (
            f"{FEATURE_INDEX_DIR_PREFIX}{cache_identity[:12]}-{uuid4().hex[:8]}"
        )
        index_dir = settings.cache_dir / index_name
        index = cls._build(catalog, settings)
        index._save(index_dir, cache_identity)
        cls._save_manifest(
            manifest_path,
            {**metadata, "feature_index_dir": index_name},
        )
        del index
        cls._cleanup_index_dirs(settings.cache_dir, index_name)
        return cls._load(
            index_dir,
            expected_image_ids,
            cache_identity,
            settings,
            loaded_from_cache=False,
        )

    @classmethod
    def _build(cls, catalog: ImageCatalog, settings: Settings) -> "FeatureIndex":
        sift_create = getattr(cv2, "SIFT_create")
        sift = sift_create(
            settings.sift_features,
            3,
            settings.sift_contrast_threshold,
            10,
            1.6,
            cv2.CV_8U,
            False,
        )
        image_ids: list[str] = []
        point_groups: list[np.ndarray] = []
        descriptor_groups: list[np.ndarray] = []
        working_widths: list[int] = []
        working_heights: list[int] = []
        working_scales: list[float] = []
        coarse_levels: list[np.ndarray] = []
        coarse_image_indices: list[int] = []
        coarse_region_sizes: list[int] = []
        if settings.coarse_template_edge <= 0:
            raise ValueError("coarse_template_edge must be positive")

        for image_index, record in enumerate(catalog.records):
            safe_record = catalog.get(record.image_id)
            image, working_scale = resize_to_max(
                read_image(safe_record.path, settings.max_image_pixels),
                settings.working_max_edge,
            )
            gray = to_gray(image)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is None:
                points = np.empty((0, 2), dtype=np.float32)
                descriptors = np.empty((0, 128), dtype=np.uint8)
            else:
                points = np.ascontiguousarray(
                    np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(
                        -1, 2
                    )
                )
                descriptors = np.ascontiguousarray(descriptors, dtype=np.uint8).reshape(-1, 128)

            image_ids.append(record.image_id)
            point_groups.append(points)
            descriptor_groups.append(descriptors)
            working_widths.append(image.shape[1])
            working_heights.append(image.shape[0])
            working_scales.append(working_scale)

            height, width = gray.shape[:2]
            shorter_edge = min(width, height)
            for region_size in cls._region_sizes(shorter_edge, settings.tile_sizes):
                level_scale = settings.coarse_template_edge / region_size
                coarse_width = max(1, round(width * level_scale))
                coarse_height = max(1, round(height * level_scale))
                interpolation = cv2.INTER_AREA if level_scale < 1.0 else cv2.INTER_CUBIC
                level = cv2.resize(
                    gray,
                    (coarse_width, coarse_height),
                    interpolation=interpolation,
                )
                coarse_levels.append(np.ascontiguousarray(level, dtype=np.uint8))
                coarse_image_indices.append(image_index)
                coarse_region_sizes.append(region_size)

        points = cls._concatenate_rows(point_groups, 2, np.float32)
        descriptors = cls._concatenate_rows(descriptor_groups, 128, np.uint8)
        point_offsets = cls._offsets(point_groups)
        descriptor_offsets = cls._offsets(descriptor_groups)
        representative_descriptors, representative_image_indices = (
            cls._build_representatives(
                points,
                descriptors,
                descriptor_offsets,
                np.asarray(working_widths, dtype=np.int32),
                np.asarray(working_heights, dtype=np.int32),
            )
        )
        return cls(
            image_ids=tuple(image_ids),
            points=points,
            point_offsets=point_offsets,
            descriptors=descriptors,
            descriptor_offsets=descriptor_offsets,
            working_widths=np.asarray(working_widths, dtype=np.int32),
            working_heights=np.asarray(working_heights, dtype=np.int32),
            working_scales=np.asarray(working_scales, dtype=np.float64),
            coarse_templates=cls._coarse_templates(
                coarse_levels,
                coarse_image_indices,
                coarse_region_sizes,
            ),
            representative_descriptors=representative_descriptors,
            representative_image_indices=representative_image_indices,
            loaded_from_cache=False,
        )

    @staticmethod
    def _build_representatives(
        points: np.ndarray,
        descriptors: np.ndarray,
        descriptor_offsets: np.ndarray,
        working_widths: np.ndarray,
        working_heights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        groups: list[np.ndarray] = []
        owner_groups: list[np.ndarray] = []
        per_cell = max(
            1,
            REPRESENTATIVES_PER_IMAGE
            // (REPRESENTATIVE_GRID_SIZE * REPRESENTATIVE_GRID_SIZE),
        )
        for image_index, (start, end) in enumerate(
            pairwise(descriptor_offsets)
        ):
            start = int(start)
            end = int(end)
            descriptor_count = end - start
            if descriptor_count <= 0:
                continue
            representative_count = min(
                REPRESENTATIVES_PER_IMAGE,
                descriptor_count,
            )
            image_points = points[start:end]
            cell_x = np.clip(
                (
                    image_points[:, 0]
                    / max(1, int(working_widths[image_index]))
                    * REPRESENTATIVE_GRID_SIZE
                ).astype(np.int32),
                0,
                REPRESENTATIVE_GRID_SIZE - 1,
            )
            cell_y = np.clip(
                (
                    image_points[:, 1]
                    / max(1, int(working_heights[image_index]))
                    * REPRESENTATIVE_GRID_SIZE
                ).astype(np.int32),
                0,
                REPRESENTATIVE_GRID_SIZE - 1,
            )
            cells = cell_y * REPRESENTATIVE_GRID_SIZE + cell_x
            selected: list[int] = []
            for cell in range(
                REPRESENTATIVE_GRID_SIZE * REPRESENTATIVE_GRID_SIZE
            ):
                cell_indices = np.flatnonzero(cells == cell)
                if len(cell_indices):
                    selected.extend(
                        cell_indices[
                            np.linspace(
                                0,
                                len(cell_indices) - 1,
                                min(per_cell, len(cell_indices)),
                                dtype=np.int64,
                            )
                        ].tolist()
                    )
            if len(selected) < representative_count:
                selected_mask = np.zeros(descriptor_count, dtype=bool)
                selected_mask[selected] = True
                remaining = np.flatnonzero(~selected_mask)
                needed = representative_count - len(selected)
                selected.extend(
                    remaining[
                        np.linspace(
                            0,
                            len(remaining) - 1,
                            needed,
                            dtype=np.int64,
                        )
                    ].tolist()
                )
            indices = np.asarray(selected[:representative_count], dtype=np.int64) + start
            groups.append(
                np.ascontiguousarray(descriptors[indices], dtype=np.float32)
            )
            owner_groups.append(
                np.full(representative_count, image_index, dtype=np.int32)
            )
        if not groups:
            return (
                np.empty((0, 128), dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )
        return (
            np.ascontiguousarray(np.concatenate(groups), dtype=np.float32),
            np.concatenate(owner_groups).astype(np.int32, copy=False),
        )

    def rank_image_indices(self, descriptors: np.ndarray) -> np.ndarray:
        image_count = len(self.image_ids)
        if image_count == 0:
            return np.empty(0, dtype=np.int32)
        k = min(5, len(self.representative_descriptors))
        if not len(descriptors) or self._global_matcher is None or k < 2:
            return np.arange(image_count, dtype=np.int32)
        query = np.ascontiguousarray(descriptors, dtype=np.float32)
        matches_by_descriptor = self._global_matcher.knnMatch(query, k=k)
        votes: dict[int, float] = defaultdict(float)
        for matches in matches_by_descriptor:
            voted_owners: set[int] = set()
            for rank, (match, next_match) in enumerate(
                zip(matches, matches[1:])
            ):
                if (
                    not np.isfinite(match.distance)
                    or not np.isfinite(next_match.distance)
                    or next_match.distance <= 0.0
                    or match.distance >= 0.78 * next_match.distance
                ):
                    continue
                descriptor_index = int(match.trainIdx)
                if not 0 <= descriptor_index < len(
                    self.representative_image_indices
                ):
                    continue
                image_index = int(
                    self.representative_image_indices[descriptor_index]
                )
                if image_index in voted_owners:
                    continue
                ratio = max(0.0, match.distance) / next_match.distance
                votes[image_index] += (1.0 - ratio) / (rank + 1)
                voted_owners.add(image_index)

        ranked_indices = sorted(
            votes,
            key=lambda image_index: (-votes[image_index], image_index),
        )
        ranked_set = set(ranked_indices)
        ranked_indices.extend(
            image_index
            for image_index in range(image_count)
            if image_index not in ranked_set
        )
        return np.asarray(ranked_indices, dtype=np.int32)

    def _save(self, index_dir: Path, cache_identity: str) -> None:
        id_width = max((len(image_id) for image_id in self.image_ids), default=1)
        arrays = {
            "cache_identity": np.asarray(cache_identity, dtype=f"<U{len(cache_identity)}"),
            "image_ids": np.asarray(self.image_ids, dtype=f"<U{id_width}"),
            "points": self.points,
            "point_offsets": self.point_offsets,
            "descriptors": self.descriptors,
            "descriptor_offsets": self.descriptor_offsets,
            "working_widths": self.working_widths,
            "working_heights": self.working_heights,
            "working_scales": self.working_scales,
            "coarse_pixels": self.coarse_templates.pixels,
            "coarse_offsets": self.coarse_templates.offsets,
            "coarse_widths": self.coarse_templates.widths,
            "coarse_heights": self.coarse_templates.heights,
            "coarse_image_indices": self.coarse_templates.image_indices,
            "coarse_region_sizes": self.coarse_templates.region_sizes,
            "representative_descriptors": self.representative_descriptors,
            "representative_image_indices": self.representative_image_indices,
        }
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f"{index_dir.name}.tmp-", dir=index_dir.parent)
        )
        try:
            for name, array in arrays.items():
                np.save(temporary_dir / f"{name}.npy", array, allow_pickle=False)
            temporary_dir.replace(index_dir)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @staticmethod
    def _cleanup_index_dirs(cache_dir: Path, keep_name: str) -> None:
        for path in cache_dir.glob(f"{FEATURE_INDEX_DIR_PREFIX}*"):
            if path.name != keep_name:
                shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(cache_dir / "feature-index", ignore_errors=True)
        (cache_dir / "features.npz").unlink(missing_ok=True)

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
        index_dir: Path,
        expected_image_ids: tuple[str, ...],
        expected_cache_identity: str,
        settings: Settings,
        *,
        loaded_from_cache: bool,
    ) -> "FeatureIndex":
        arrays: dict[str, np.ndarray] = {}
        for name in _ARRAY_NAMES:
            path = index_dir / f"{name}.npy"
            arrays[name] = np.load(
                path,
                mmap_mode="r" if name in _MMAP_ARRAYS else None,
                allow_pickle=False,
            )
        cls._validate_arrays(arrays, expected_image_ids, expected_cache_identity, settings)
        image_ids = tuple(str(image_id) for image_id in arrays["image_ids"])
        coarse_templates = CoarseTemplateFeatures(
            pixels=arrays["coarse_pixels"],
            offsets=arrays["coarse_offsets"],
            widths=arrays["coarse_widths"],
            heights=arrays["coarse_heights"],
            image_indices=arrays["coarse_image_indices"],
            region_sizes=arrays["coarse_region_sizes"],
        )
        return cls(
            image_ids=image_ids,
            points=arrays["points"],
            point_offsets=arrays["point_offsets"],
            descriptors=arrays["descriptors"],
            descriptor_offsets=arrays["descriptor_offsets"],
            working_widths=arrays["working_widths"],
            working_heights=arrays["working_heights"],
            working_scales=arrays["working_scales"],
            coarse_templates=coarse_templates,
            representative_descriptors=arrays["representative_descriptors"],
            representative_image_indices=arrays["representative_image_indices"],
            loaded_from_cache=loaded_from_cache,
        )

    @staticmethod
    def _validate_arrays(
        arrays: dict[str, np.ndarray],
        expected_image_ids: tuple[str, ...],
        expected_cache_identity: str,
        settings: Settings,
    ) -> None:
        expected_dtypes = {
            "points": np.dtype(np.float32),
            "point_offsets": np.dtype(np.int64),
            "descriptors": np.dtype(np.uint8),
            "descriptor_offsets": np.dtype(np.int64),
            "working_widths": np.dtype(np.int32),
            "working_heights": np.dtype(np.int32),
            "working_scales": np.dtype(np.float64),
            "coarse_pixels": np.dtype(np.uint8),
            "coarse_offsets": np.dtype(np.int64),
            "coarse_widths": np.dtype(np.int32),
            "coarse_heights": np.dtype(np.int32),
            "coarse_image_indices": np.dtype(np.int32),
            "coarse_region_sizes": np.dtype(np.int32),
            "representative_descriptors": np.dtype(np.float32),
            "representative_image_indices": np.dtype(np.int32),
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

        representative_descriptors = arrays["representative_descriptors"]
        representative_image_indices = arrays[
            "representative_image_indices"
        ]
        if (
            representative_descriptors.ndim != 2
            or representative_descriptors.shape[1] != 128
        ):
            raise ValueError("Invalid cached representative descriptors")
        if representative_image_indices.shape != (
            len(representative_descriptors),
        ):
            raise ValueError("Invalid cached representative owners")
        if len(representative_image_indices) and (
            representative_image_indices.min() < 0
            or representative_image_indices.max() >= image_count
        ):
            raise ValueError("Cached representative owner is out of range")

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
        if len(coarse_owners) and (
            coarse_owners.min() < 0 or coarse_owners.max() >= image_count
        ):
            raise ValueError("Cached coarse template owner is out of range")
        if image_count and not np.array_equal(
            np.unique(coarse_owners), np.arange(image_count, dtype=np.int32)
        ):
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
            for region_size in FeatureIndex._region_sizes(
                min(width, height), settings.tile_sizes
            ):
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
            and np.array_equal(
                arrays["coarse_widths"], np.asarray(expected_widths, dtype=np.int32)
            )
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
        if not region_sizes and shorter_edge > 0:
            region_sizes.add(shorter_edge)
        return sorted(region_sizes)

    @staticmethod
    def _concatenate_rows(groups: list[np.ndarray], width: int, dtype: Any) -> np.ndarray:
        if not groups:
            return np.empty((0, width), dtype=dtype)
        return np.ascontiguousarray(
            np.concatenate(groups, axis=0),
            dtype=dtype,
        ).reshape(-1, width)

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
