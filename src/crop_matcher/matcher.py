from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import (
    gradient_magnitude,
    normalized_correlation,
    read_image,
    resize_to_max,
    to_gray,
)

_ALIGNMENT_SCALE_FACTORS = (
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.125,
    1.15,
    1.2,
)
_ALIGNMENT_SEARCH_RADIUS = 12
_AXIS_RANSAC_REPROJECTION_THRESHOLD = 4.0
_AXIS_RANSAC_MAX_MODELS = 256
_AXIS_RANSAC_MIN_BASELINE_SQUARED = 16.0
_STRUCTURE_GRADIENT_THRESHOLD = 64.0
_STRUCTURE_DENSITY_SATURATION = 0.35
_GEOMETRY_WEIGHT_MIN = 0.1
_GEOMETRY_WEIGHT_MAX = 0.35
_GEOMETRY_COUNT_SATURATION = 12
_GEOMETRY_SPREAD_SATURATION = 0.25
_GEOMETRY_MIN_AXIS_SPAN = 0.02
_EDGE_WEIGHT_MIN = 0.1
_EDGE_WEIGHT_MAX = 0.3
_SPARSE_TEMPLATE_REFINEMENT_RELIABILITY = 0.5
_GEOMETRY_QUALITY_FLOOR = 0.5
_GEOMETRY_QUALITY_RANGE = 0.4
_MIN_TEMPLATE_REFINEMENT_COUNT = 1


class _NoMatchEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MatchResult:
    record: ImageRecord
    similarity: float
    method: str
    inlier_count: int
    inlier_ratio: float
    appearance_score: float


@dataclass(frozen=True, slots=True)
class CandidateScore:
    record: ImageRecord
    appearance: float
    geometry_quality: float
    score: float
    inlier_count: int
    inlier_ratio: float


class ImageMatcher:
    def __init__(self, catalog: ImageCatalog, index: FeatureIndex, settings: Settings) -> None:
        self.catalog = catalog
        self.index = index
        self.settings = settings
        self._flann_lock = Lock()
        self._sift_lock = Lock()
        self._sift = cv2.SIFT_create(
            nfeatures=settings.sift_features,
            contrastThreshold=settings.sift_contrast_threshold,
        )

    def match(self, query_bgr: np.ndarray) -> MatchResult:
        return self.match_many(query_bgr, limit=1)[0]

    def match_many(self, query_bgr: np.ndarray, limit: int = 3) -> list[MatchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")

        query_gray = to_gray(query_bgr)
        target_count = min(limit, len(self.catalog.records))
        feature_query, extraction_scale = self._feature_query(query_gray)
        with self._sift_lock:
            keypoints, descriptors = self._sift.detectAndCompute(feature_query, None)
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        primary, coarse_candidate_indices = (
            self._primary_results(
                query_gray,
                points,
                descriptors,
                extraction_scale,
                target_count,
            )
            if descriptors is not None and len(descriptors) >= 4
            else ([], None)
        )
        merged = {result.record.image_id: result for result in primary}
        tiny_query = bool(self.settings.tile_sizes) and min(query_gray.shape[:2]) < min(
            self.settings.tile_sizes
        )
        if (tiny_query and target_count > 0) or len(merged) < target_count:
            exclude_ids = set() if tiny_query else set(merged)
            requested_count = target_count if tiny_query else target_count - len(merged)
            if not tiny_query and coarse_candidate_indices is not None:
                fallback = self._fallback_results(
                    query_gray,
                    exclude_ids,
                    requested_count,
                    coarse_candidate_indices,
                )
            else:
                fallback = self._fallback_results(query_gray, exclude_ids, requested_count)
            for result in fallback:
                existing = merged.get(result.record.image_id)
                if existing is None or result.similarity > existing.similarity:
                    merged[result.record.image_id] = result
        return sorted(
            merged.values(),
            key=lambda result: (-result.similarity, result.record.image_id),
        )[:target_count]

    def _primary_results(
        self,
        query_gray: np.ndarray,
        query_points: np.ndarray,
        query_descriptors: np.ndarray,
        extraction_scale: float,
        target_count: int,
    ) -> tuple[list[MatchResult], list[int] | None]:
        query_gradient = gradient_magnitude(query_gray)
        structure_reliability = self._query_structure_reliability(query_gradient)
        geometry_weight = self._query_geometry_weight(query_gradient)
        retrieved_ids = self._retrieve(query_descriptors)
        candidates = [
            score
            for image_id in retrieved_ids
            if (
                score := self._verify(
                    image_id,
                    query_gray,
                    query_points,
                    query_descriptors,
                    extraction_scale,
                    query_gradient,
                    geometry_weight,
                    structure_reliability,
                )
            )
            is not None
        ]
        desired_count = min(target_count, len(self.catalog.records))
        coarse_candidate_indices: list[int] | None = None
        if len(candidates) < desired_count:
            candidate_ids = {candidate.record.image_id for candidate in candidates}
            coarse_candidate_indices = self._coarse_candidates(
                query_gray,
                candidate_ids,
                desired_count - len(candidates),
            )
            retrieved_set = set(retrieved_ids)
            for index in coarse_candidate_indices:
                image_id = self.index.image_ids[index]
                if image_id in retrieved_set:
                    continue
                score = self._verify(
                    image_id,
                    query_gray,
                    query_points,
                    query_descriptors,
                    extraction_scale,
                    query_gradient,
                    geometry_weight,
                    structure_reliability,
                )
                if score is not None:
                    candidates.append(score)
                    if len(candidates) >= desired_count:
                        break
        if structure_reliability < _SPARSE_TEMPLATE_REFINEMENT_RELIABILITY:
            image_indices = {
                image_id: index for index, image_id in enumerate(self.index.image_ids)
            }
            rescored: list[CandidateScore] = []
            for candidate in candidates:
                template_appearance = self._template_score(
                    image_indices[candidate.record.image_id],
                    query_gray,
                    query_gradient,
                    structure_reliability,
                )
                if template_appearance > candidate.score:
                    candidate = CandidateScore(
                        candidate.record,
                        template_appearance,
                        candidate.geometry_quality,
                        template_appearance,
                        candidate.inlier_count,
                        candidate.inlier_ratio,
                    )
                rescored.append(candidate)
            candidates = rescored
        candidates.sort(key=lambda item: (-item.score, item.record.image_id))
        results = [
            MatchResult(
                candidate.record,
                round(100.0 * candidate.score, 1),
                "sift",
                candidate.inlier_count,
                candidate.inlier_ratio,
                candidate.appearance,
            )
            for candidate in candidates
        ]
        return results, coarse_candidate_indices

    def _feature_query(self, query_gray: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = query_gray.shape[:2]
        shortest_edge = min(height, width)
        if shortest_edge >= self.settings.query_feature_min_edge:
            return query_gray.copy(), 1.0
        scale = self.settings.query_feature_min_edge / shortest_edge
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(query_gray, size, interpolation=cv2.INTER_CUBIC), scale

    def _retrieve(self, descriptors: np.ndarray) -> list[str]:
        target_count = min(self.settings.candidate_count, len(self.index.image_ids))
        matches_by_descriptor = []
        k = min(5, len(self.index.descriptors))
        if k >= 2:
            with self._flann_lock:
                matches_by_descriptor = self.index.global_matcher.knnMatch(descriptors, k=k)

        votes: dict[int, float] = defaultdict(float)
        for matches in matches_by_descriptor:
            voted_owners: set[int] = set()
            for rank, (match, next_match) in enumerate(zip(matches, matches[1:])):
                if (
                    not np.isfinite(match.distance)
                    or not np.isfinite(next_match.distance)
                    or next_match.distance <= 0.0
                    or match.distance >= 0.78 * next_match.distance
                ):
                    continue
                descriptor_index = int(match.trainIdx)
                if not 0 <= descriptor_index < len(self.index.descriptor_image_indices):
                    continue
                image_index = int(self.index.descriptor_image_indices[descriptor_index])
                if not 0 <= image_index < len(self.index.image_ids) or image_index in voted_owners:
                    continue
                ratio = max(0.0, match.distance) / next_match.distance
                votes[image_index] += (1.0 - ratio) / (rank + 1)
                voted_owners.add(image_index)

        ranked_indices = sorted(votes, key=lambda index: (-votes[index], index))
        ranked_set = set(ranked_indices)
        ranked_indices.extend(
            image_index
            for image_index in range(len(self.index.image_ids))
            if image_index not in ranked_set
        )
        return [self.index.image_ids[index] for index in ranked_indices[:target_count]]

    def _verify(
        self,
        image_id: str,
        query_gray: np.ndarray,
        query_points: np.ndarray,
        query_descriptors: np.ndarray,
        extraction_scale: float,
        query_gradient: np.ndarray | None = None,
        geometry_weight: float | None = None,
        structure_reliability: float | None = None,
    ) -> CandidateScore | None:
        candidate_features = self.index.by_image[image_id]
        if len(candidate_features.descriptors) < 2:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            query_descriptors,
            candidate_features.descriptors,
            k=2,
        )
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.78 * second.distance
        ]
        if len(good) < 4:
            return None

        source_points = np.asarray(
            [query_points[match.queryIdx] for match in good], dtype=np.float32
        )
        destination_points = np.asarray(
            [candidate_features.points[match.trainIdx] for match in good], dtype=np.float32
        )
        feature_affine, inlier_mask = self._estimate_axis_aligned_transform(
            source_points,
            destination_points,
        )
        if feature_affine is None or inlier_mask is None:
            return None

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 4:
            return None
        inlier_ratio = inlier_count / len(good)

        query_scale = np.array(
            [[extraction_scale, 0.0, 0.0], [0.0, extraction_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        feature_homogeneous = np.vstack([feature_affine, [0.0, 0.0, 1.0]])
        query_to_candidate = (feature_homogeneous @ query_scale)[:2]
        query_to_candidate /= candidate_features.working_scale
        if not np.isfinite(query_to_candidate).all():
            return None

        affine_scale = float(np.hypot(query_to_candidate[0, 0], query_to_candidate[0, 1]))
        if affine_scale < 0.05:
            return None

        record = self.catalog.get(image_id)
        query_height, query_width = query_gray.shape[:2]
        corners = np.asarray(
            [
                [0.0, 0.0],
                [query_width - 1.0, 0.0],
                [query_width - 1.0, query_height - 1.0],
                [0.0, query_height - 1.0],
            ],
            dtype=np.float64,
        )
        mapped_corners = cv2.transform(corners[None, :, :], query_to_candidate)[0]
        if not self._mapped_geometry_is_valid(mapped_corners, record.width, record.height):
            return None

        candidate_gray = to_gray(read_image(record.path, self.settings.max_image_pixels))
        warped = self._refined_warp(query_gray, candidate_gray, query_to_candidate)
        if query_gradient is None:
            query_gradient = gradient_magnitude(query_gray)
        if structure_reliability is None:
            structure_reliability = self._query_structure_reliability(query_gradient)
        if geometry_weight is None:
            geometry_weight = self._query_geometry_weight(query_gradient)
        gray_score = self._unit_correlation(query_gray, warped)
        edge_score = self._unit_correlation(
            query_gradient,
            gradient_magnitude(warped),
        )
        appearance = self._appearance_score(
            gray_score,
            edge_score,
            structure_reliability,
        )
        feature_shape = (
            max(1, round(query_height * extraction_scale)),
            max(1, round(query_width * extraction_scale)),
        )
        geometry_quality = self._geometry_quality(
            source_points,
            destination_points,
            feature_affine,
            inliers,
            feature_shape,
            inlier_ratio,
        )
        score = self._adaptive_score(appearance, geometry_quality, geometry_weight)
        return CandidateScore(
            record,
            appearance,
            geometry_quality,
            score,
            inlier_count,
            inlier_ratio,
        )

    @staticmethod
    def _query_structure_reliability(query_gradient: np.ndarray) -> float:
        if not query_gradient.size:
            return 0.0
        density = np.count_nonzero(
            query_gradient >= _STRUCTURE_GRADIENT_THRESHOLD
        ) / query_gradient.size
        return float(np.clip(density / _STRUCTURE_DENSITY_SATURATION, 0.0, 1.0))

    @classmethod
    def _query_geometry_weight(cls, query_gradient: np.ndarray) -> float:
        structure_reliability = cls._query_structure_reliability(query_gradient)
        return _GEOMETRY_WEIGHT_MAX - (
            (_GEOMETRY_WEIGHT_MAX - _GEOMETRY_WEIGHT_MIN) * structure_reliability
        )

    @staticmethod
    def _appearance_score(
        gray_score: float,
        edge_score: float,
        structure_reliability: float,
    ) -> float:
        edge_weight = _EDGE_WEIGHT_MIN + (
            (_EDGE_WEIGHT_MAX - _EDGE_WEIGHT_MIN) * structure_reliability
        )
        return (1.0 - edge_weight) * gray_score + edge_weight * edge_score

    @staticmethod
    def _estimate_axis_aligned_transform(
        source_points: np.ndarray,
        destination_points: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        point_count = len(source_points)
        if (
            source_points.shape != destination_points.shape
            or source_points.ndim != 2
            or source_points.shape[1:] != (2,)
            or point_count < 2
        ):
            return None, None

        source = source_points.astype(np.float64, copy=False)
        destination = destination_points.astype(np.float64, copy=False)
        first_indices, second_indices = np.triu_indices(point_count, k=1)
        source_deltas = source[second_indices] - source[first_indices]
        destination_deltas = destination[second_indices] - destination[first_indices]
        baselines = np.einsum("ij,ij->i", source_deltas, source_deltas)
        scales = np.einsum("ij,ij->i", source_deltas, destination_deltas) / np.maximum(
            baselines,
            1.0,
        )
        valid = (
            (baselines >= _AXIS_RANSAC_MIN_BASELINE_SQUARED)
            & np.isfinite(scales)
            & (scales > 0.0)
        )
        if not np.any(valid):
            return None, None

        first_indices = first_indices[valid]
        second_indices = second_indices[valid]
        baselines = baselines[valid]
        scales = scales[valid]
        if len(scales) > _AXIS_RANSAC_MAX_MODELS:
            selected = np.argpartition(
                baselines,
                -_AXIS_RANSAC_MAX_MODELS,
            )[-_AXIS_RANSAC_MAX_MODELS:]
            first_indices = first_indices[selected]
            second_indices = second_indices[selected]
            scales = scales[selected]

        translations = 0.5 * (
            destination[first_indices]
            - scales[:, None] * source[first_indices]
            + destination[second_indices]
            - scales[:, None] * source[second_indices]
        )
        projected = scales[:, None, None] * source[None, :, :] + translations[:, None, :]
        errors = np.linalg.norm(projected - destination[None, :, :], axis=2)
        model_inliers = errors <= _AXIS_RANSAC_REPROJECTION_THRESHOLD
        inlier_counts = model_inliers.sum(axis=1)
        mean_errors = np.divide(
            np.where(model_inliers, errors, 0.0).sum(axis=1),
            inlier_counts,
            out=np.full(len(inlier_counts), np.inf),
            where=inlier_counts > 0,
        )
        best_index = int(np.lexsort((mean_errors, -inlier_counts))[0])
        inliers = model_inliers[best_index]
        scale = float(scales[best_index])
        translation = translations[best_index]

        for _ in range(2):
            inlier_source = source[inliers]
            inlier_destination = destination[inliers]
            if len(inlier_source) < 2:
                return None, None
            source_mean = inlier_source.mean(axis=0)
            destination_mean = inlier_destination.mean(axis=0)
            centered_source = inlier_source - source_mean
            denominator = float(np.einsum("ij,ij->", centered_source, centered_source))
            if denominator <= 0.0:
                return None, None
            scale = float(
                np.einsum(
                    "ij,ij->",
                    centered_source,
                    inlier_destination - destination_mean,
                )
                / denominator
            )
            if not np.isfinite(scale) or scale <= 0.0:
                return None, None
            translation = destination_mean - scale * source_mean
            errors = np.linalg.norm(scale * source + translation - destination, axis=1)
            updated_inliers = errors <= _AXIS_RANSAC_REPROJECTION_THRESHOLD
            if np.array_equal(updated_inliers, inliers):
                break
            inliers = updated_inliers

        affine = np.asarray(
            [
                [scale, 0.0, translation[0]],
                [0.0, scale, translation[1]],
            ],
            dtype=np.float64,
        )
        return affine, inliers.astype(np.uint8)[:, None]

    @staticmethod
    def _geometry_quality(
        source_points: np.ndarray,
        destination_points: np.ndarray,
        affine: np.ndarray,
        inliers: np.ndarray,
        feature_shape: tuple[int, int],
        inlier_ratio: float,
    ) -> float:
        inlier_count = int(inliers.sum())
        projected = cv2.transform(source_points[None, :, :], affine)[0]
        errors = np.linalg.norm(
            projected[inliers] - destination_points[inliers],
            axis=1,
        )
        reprojection_quality = float(np.clip(1.0 - errors.mean() / 4.0, 0.0, 1.0))
        count_quality = min(inlier_count / _GEOMETRY_COUNT_SATURATION, 1.0)

        inlier_points = source_points[inliers]
        feature_height, feature_width = feature_shape
        x_span = float(np.ptp(inlier_points[:, 0]) / max(1, feature_width - 1))
        y_span = float(np.ptp(inlier_points[:, 1]) / max(1, feature_height - 1))
        spread_quality = float(
            np.sqrt(
                np.clip(x_span / _GEOMETRY_SPREAD_SATURATION, 0.0, 1.0)
                * np.clip(y_span / _GEOMETRY_SPREAD_SATURATION, 0.0, 1.0)
            )
        )
        quality = float(
            np.clip(
                0.5 * inlier_ratio
                + 0.1 * count_quality
                + 0.25 * reprojection_quality
                + 0.15 * spread_quality,
                0.0,
                1.0,
            )
        )
        if min(x_span, y_span) < _GEOMETRY_MIN_AXIS_SPAN:
            return min(quality, _GEOMETRY_QUALITY_FLOOR)
        return quality

    @staticmethod
    def _adaptive_score(
        appearance: float,
        geometry_quality: float,
        geometry_weight: float,
    ) -> float:
        geometry_confidence = float(
            np.clip(
                (geometry_quality - _GEOMETRY_QUALITY_FLOOR) / _GEOMETRY_QUALITY_RANGE,
                0.0,
                1.0,
            )
        )
        geometry_signal = geometry_confidence
        factor = 1.0 + 2.0 * geometry_weight * geometry_signal * (1.0 - appearance)
        return float(np.clip(appearance * factor, 0.0, 1.0))

    @staticmethod
    def _refined_warp(
        query_gray: np.ndarray,
        candidate_gray: np.ndarray,
        initial_transform: np.ndarray,
    ) -> np.ndarray:
        query_height, query_width = query_gray.shape[:2]
        query_center = np.asarray(
            [(query_width - 1.0) / 2.0, (query_height - 1.0) / 2.0, 1.0],
            dtype=np.float64,
        )
        candidate_center = initial_transform @ query_center
        initial_scale = float(np.hypot(initial_transform[0, 0], initial_transform[0, 1]))
        best_warp = cv2.warpAffine(
            candidate_gray,
            initial_transform,
            (query_width, query_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        best_score = normalized_correlation(query_gray, best_warp)
        if best_score >= 0.9:
            return best_warp

        for factor in _ALIGNMENT_SCALE_FACTORS:
            scale = initial_scale * factor
            half_width = scale * (query_width / 2.0 + _ALIGNMENT_SEARCH_RADIUS + 2)
            half_height = scale * (query_height / 2.0 + _ALIGNMENT_SEARCH_RADIUS + 2)
            left = max(0, int(np.floor(candidate_center[0] - half_width)))
            top = max(0, int(np.floor(candidate_center[1] - half_height)))
            right = min(candidate_gray.shape[1], int(np.ceil(candidate_center[0] + half_width)))
            bottom = min(
                candidate_gray.shape[0], int(np.ceil(candidate_center[1] + half_height))
            )
            region = candidate_gray[top:bottom, left:right]
            if (
                region.shape[1] < round(scale * query_width)
                or region.shape[0] < round(scale * query_height)
            ):
                continue

            resized_width = max(query_width, round(region.shape[1] / scale))
            resized_height = max(query_height, round(region.shape[0] / scale))
            resized = cv2.resize(
                region,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA if scale > 1.0 else cv2.INTER_CUBIC,
            )
            response = cv2.matchTemplate(resized, query_gray, cv2.TM_CCOEFF_NORMED)
            response = np.nan_to_num(
                response,
                copy=False,
                nan=-1.0,
                posinf=-1.0,
                neginf=-1.0,
            )
            _, _, _, location = cv2.minMaxLoc(response)
            scale_x = region.shape[1] / resized_width
            scale_y = region.shape[0] / resized_height
            candidate_transform = np.asarray(
                [
                    [scale_x, 0.0, left + location[0] * scale_x],
                    [0.0, scale_y, top + location[1] * scale_y],
                ],
                dtype=np.float64,
            )
            candidate_warp = cv2.warpAffine(
                candidate_gray,
                candidate_transform,
                (query_width, query_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
            score = normalized_correlation(query_gray, candidate_warp)
            if score > best_score:
                best_score = score
                best_warp = candidate_warp

        return best_warp


    @staticmethod
    def _mapped_geometry_is_valid(
        mapped_corners: np.ndarray, candidate_width: int, candidate_height: int
    ) -> bool:
        if mapped_corners.shape != (4, 2) or not np.isfinite(mapped_corners).all():
            return False
        x_tolerance = 0.05 * candidate_width
        y_tolerance = 0.05 * candidate_height
        xs = mapped_corners[:, 0]
        ys = mapped_corners[:, 1]
        return bool(
            np.all(xs >= -x_tolerance)
            and np.all(xs <= candidate_width - 1 + x_tolerance)
            and np.all(ys >= -y_tolerance)
            and np.all(ys <= candidate_height - 1 + y_tolerance)
            and abs(cv2.contourArea(mapped_corners.astype(np.float32))) > 1.0
        )

    @staticmethod
    def _unit_correlation(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.clip(normalized_correlation(left, right), 0.0, 1.0))

    def _fallback(self, query_gray: np.ndarray) -> MatchResult:
        return self._fallback_results(query_gray, set())[0]

    def _fallback_results(
        self,
        query_gray: np.ndarray,
        exclude_ids: set[str],
        requested_count: int = 0,
        candidate_indices: list[int] | None = None,
    ) -> list[MatchResult]:
        if (
            not self.catalog.records
            or not self.index.image_ids
            or not len(self.index.coarse_templates.pixels)
        ):
            raise _NoMatchEvidenceError("Fallback index is empty")

        if candidate_indices is None:
            candidate_indices = self._coarse_candidates(
                query_gray,
                exclude_ids,
                requested_count,
            )
        else:
            candidate_indices = [
                index
                for index in candidate_indices
                if self.index.image_ids[index] not in exclude_ids
            ]
        refinement_count = min(
            max(requested_count, _MIN_TEMPLATE_REFINEMENT_COUNT),
            len(candidate_indices),
        )
        candidate_indices = candidate_indices[:refinement_count]
        query_gradient = gradient_magnitude(query_gray)
        structure_reliability = self._query_structure_reliability(query_gradient)
        scored = [
            self._template_score(
                index,
                query_gray,
                query_gradient,
                structure_reliability,
            )
            for index in candidate_indices
        ]
        ranking = sorted(
            zip(candidate_indices, scored, strict=True),
            key=lambda item: (-item[1], self.index.image_ids[item[0]]),
        )
        results = [
            MatchResult(
                self.catalog.get(self.index.image_ids[image_index]),
                round(100.0 * score, 1),
                "template",
                0,
                0.0,
                score,
            )
            for image_index, score in ranking
        ]
        return sorted(results, key=lambda result: (-result.similarity, result.record.image_id))

    def _coarse_candidates(
        self, query_gray: np.ndarray, exclude_ids: set[str], requested_count: int
    ) -> list[int]:
        available_count = sum(image_id not in exclude_ids for image_id in self.index.image_ids)
        target_count = min(max(1, self.settings.candidate_count, requested_count), available_count)
        if target_count == 0:
            return []

        query_height, query_width = query_gray.shape[:2]
        shortest_edge = min(query_height, query_width)
        if shortest_edge <= 0:
            return []
        scale = self.settings.coarse_template_edge / shortest_edge
        coarse_query = cv2.resize(
            query_gray,
            (max(1, round(query_width * scale)), max(1, round(query_height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        coarse = self.index.coarse_templates
        scores: dict[int, float] = {}
        level_count = len(coarse.image_indices)
        for level_index in range(level_count):
            if (
                level_index >= len(coarse.widths)
                or level_index >= len(coarse.heights)
                or level_index + 1 >= len(coarse.offsets)
            ):
                continue
            image_index = int(coarse.image_indices[level_index])
            if (
                not 0 <= image_index < len(self.index.image_ids)
                or self.index.image_ids[image_index] in exclude_ids
            ):
                continue
            width = int(coarse.widths[level_index])
            height = int(coarse.heights[level_index])
            start = int(coarse.offsets[level_index])
            end = int(coarse.offsets[level_index + 1])
            if (
                width < coarse_query.shape[1]
                or height < coarse_query.shape[0]
                or width <= 0
                or height <= 0
                or start < 0
                or end < start
                or end > len(coarse.pixels)
                or end - start != width * height
            ):
                continue
            level = coarse.pixels[start:end].reshape(height, width)
            score = self._coarse_template_peak(level, coarse_query)
            if np.isfinite(score):
                scores[image_index] = max(scores.get(image_index, 0.0), score)

        ranked = sorted(scores, key=lambda index: (-scores[index], self.index.image_ids[index]))
        ranked_set = set(ranked)
        ranked.extend(
            image_index
            for image_index, image_id in enumerate(self.index.image_ids)
            if image_index not in ranked_set and image_id not in exclude_ids
        )
        return ranked[:target_count]

    @staticmethod
    def _coarse_template_peak(candidate: np.ndarray, query: np.ndarray) -> float:
        response = cv2.matchTemplate(candidate, query, cv2.TM_CCOEFF_NORMED)
        finite = response[np.isfinite(response)]
        if not len(finite):
            return float("nan")
        return float(np.clip(float(finite.max()), 0.0, 1.0))

    def _template_score(
        self,
        image_index: int,
        query_gray: np.ndarray,
        query_gradient: np.ndarray | None = None,
        structure_reliability: float | None = None,
    ) -> float:
        query_height, query_width = query_gray.shape[:2]
        if query_height == 0 or query_width == 0:
            return 0.0

        record = self.catalog.get(self.index.image_ids[image_index])
        candidate, _ = resize_to_max(
            read_image(record.path, self.settings.max_image_pixels),
            self.settings.working_max_edge,
        )
        candidate_gray = to_gray(candidate)
        candidate_height, candidate_width = candidate_gray.shape[:2]
        owner_mask = self.index.coarse_templates.image_indices == image_index
        pyramid_sizes = sorted(
            {int(size) for size in self.index.coarse_templates.region_sizes[owner_mask]}
        )
        if query_gradient is None:
            query_gradient = gradient_magnitude(query_gray)
        if structure_reliability is None:
            structure_reliability = self._query_structure_reliability(query_gradient)
        best_score = 0.0
        seen_dimensions: set[tuple[int, int]] = set()

        for tile_size in pyramid_sizes:
            if tile_size <= 0:
                continue
            scale = min(1.0, min(query_height, query_width) / int(tile_size))
            level_width = max(1, round(candidate_width * scale))
            level_height = max(1, round(candidate_height * scale))
            dimensions = (level_width, level_height)
            if dimensions in seen_dimensions:
                continue
            seen_dimensions.add(dimensions)
            if query_width > level_width or query_height > level_height:
                continue
            if dimensions == (candidate_width, candidate_height):
                level = candidate_gray
            else:
                level = cv2.resize(candidate_gray, dimensions, interpolation=cv2.INTER_AREA)

            gray_score = self._template_peak(level, query_gray)
            edge_score = self._template_peak(gradient_magnitude(level), query_gradient)
            appearance = self._appearance_score(
                gray_score,
                edge_score,
                structure_reliability,
            )
            best_score = max(best_score, appearance)

        return float(np.clip(best_score, 0.0, 1.0))

    @staticmethod
    def _template_peak(candidate: np.ndarray, query: np.ndarray) -> float:
        response = cv2.matchTemplate(candidate, query, cv2.TM_CCOEFF_NORMED)
        finite = response[np.isfinite(response)]
        peak = float(finite.max()) if len(finite) else -1.0
        return float(np.clip(peak, 0.0, 1.0))
