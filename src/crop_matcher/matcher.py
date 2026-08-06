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
    geometry: float
    appearance: float
    raw_score: float
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
        feature_query, extraction_scale = self._feature_query(query_gray)
        with self._sift_lock:
            keypoints, descriptors = self._sift.detectAndCompute(feature_query, None)
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        primary = (
            self._primary_results(
                query_gray,
                points,
                descriptors,
                extraction_scale,
            )
            if descriptors is not None and len(descriptors) >= 4
            else []
        )
        target_count = min(limit, len(self.catalog.records))
        merged = {result.record.image_id: result for result in primary}
        tiny_query = bool(self.settings.tile_sizes) and min(query_gray.shape[:2]) < min(
            self.settings.tile_sizes
        )
        if (tiny_query and target_count > 0) or len(merged) < target_count:
            exclude_ids = set() if tiny_query else set(merged)
            requested_count = target_count if tiny_query else target_count - len(merged)
            for result in self._fallback_results(query_gray, exclude_ids, requested_count):
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
    ) -> list[MatchResult]:
        candidates = [
            score
            for image_id in self._retrieve(query_descriptors)
            if (
                score := self._verify(
                    image_id,
                    query_gray,
                    query_points,
                    query_descriptors,
                    extraction_scale,
                )
            )
            is not None
        ]
        candidates.sort(key=lambda item: (-item.raw_score, item.record.image_id))
        if not candidates:
            return []

        second = candidates[1].raw_score if len(candidates) > 1 else 0.0
        margin = float(np.clip((candidates[0].raw_score - second) / 0.2, 0.0, 1.0))
        results = [
            MatchResult(
                candidate.record,
                round(100.0 * float(np.clip(candidate.raw_score + 0.1 * margin, 0.0, 1.0)), 1),
                "sift",
                candidate.inlier_count,
                candidate.inlier_ratio,
                candidate.appearance,
            )
            for candidate in candidates
        ]
        return sorted(results, key=lambda result: (-result.similarity, result.record.image_id))

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
        feature_affine, inlier_mask = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if feature_affine is None or inlier_mask is None or not np.isfinite(feature_affine).all():
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
        warped = cv2.warpAffine(
            candidate_gray,
            query_to_candidate,
            (query_width, query_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        gray_score = self._unit_correlation(query_gray, warped)
        edge_score = self._unit_correlation(
            gradient_magnitude(query_gray),
            gradient_magnitude(warped),
        )
        appearance = 0.7 * gray_score + 0.3 * edge_score

        projected = cv2.transform(source_points[None, :, :], feature_affine)[0]
        errors = np.linalg.norm(projected[inliers] - destination_points[inliers], axis=1)
        reprojection_quality = float(np.clip(1.0 - errors.mean() / 4.0, 0.0, 1.0))
        count_quality = min(inlier_count / 20.0, 1.0)
        geometry = float(
            np.clip(0.5 * inlier_ratio + 0.3 * count_quality + 0.2 * reprojection_quality, 0.0, 1.0)
        )
        raw_score = 0.4 * geometry + 0.5 * appearance
        return CandidateScore(
            record,
            geometry,
            appearance,
            raw_score,
            inlier_count,
            inlier_ratio,
        )

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
        return float(np.clip((normalized_correlation(left, right) + 1.0) / 2.0, 0.0, 1.0))

    def _fallback(self, query_gray: np.ndarray) -> MatchResult:
        return self._fallback_results(query_gray, set())[0]

    def _fallback_results(
        self,
        query_gray: np.ndarray,
        exclude_ids: set[str],
        requested_count: int = 0,
    ) -> list[MatchResult]:
        if (
            not self.catalog.records
            or not self.index.image_ids
            or not len(self.index.coarse_templates.pixels)
        ):
            raise _NoMatchEvidenceError("Fallback index is empty")

        candidate_indices = self._coarse_candidates(query_gray, exclude_ids, requested_count)
        scored = [self._template_score(index, query_gray) for index in candidate_indices]
        ranking = sorted(
            zip(candidate_indices, scored, strict=True),
            key=lambda item: (-item[1], self.index.image_ids[item[0]]),
        )
        if not ranking:
            return []
        _, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = float(np.clip((best_score - second_score) / 0.2, 0.0, 1.0))
        results = [
            MatchResult(
                self.catalog.get(self.index.image_ids[image_index]),
                round(min(89.9, 100.0 * (0.85 * score + 0.15 * margin)), 1),
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
        return float(np.clip((float(finite.max()) + 1.0) / 2.0, 0.0, 1.0))

    def _template_score(self, image_index: int, query_gray: np.ndarray) -> float:
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
        query_gradient = gradient_magnitude(query_gray)
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
            best_score = max(best_score, 0.7 * gray_score + 0.3 * edge_score)

        return float(np.clip(best_score, 0.0, 1.0))

    @staticmethod
    def _template_peak(candidate: np.ndarray, query: np.ndarray) -> float:
        response = cv2.matchTemplate(candidate, query, cv2.TM_CCOEFF_NORMED)
        finite = response[np.isfinite(response)]
        peak = float(finite.max()) if len(finite) else -1.0
        return float(np.clip((peak + 1.0) / 2.0, 0.0, 1.0))
