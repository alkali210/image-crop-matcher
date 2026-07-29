from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import gradient_magnitude, normalized_correlation, read_image, to_gray


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
        self._sift = cv2.SIFT_create(
            nfeatures=settings.sift_features,
            contrastThreshold=settings.sift_contrast_threshold,
        )

    def match(self, query_bgr: np.ndarray) -> MatchResult:
        query_gray = to_gray(query_bgr)
        feature_query, extraction_scale = self._feature_query(query_gray)
        keypoints, descriptors = self._sift.detectAndCompute(feature_query, None)
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        if descriptors is None or len(descriptors) < 4:
            return self._fallback(query_gray)

        candidate_ids = self._retrieve(descriptors)
        candidates = [
            score
            for image_id in candidate_ids
            if (
                score := self._verify(
                    image_id,
                    query_gray,
                    points,
                    descriptors,
                    extraction_scale,
                )
            )
            is not None
        ]
        if not candidates:
            return self._fallback(query_gray)

        candidates.sort(key=lambda item: item.raw_score, reverse=True)
        best = candidates[0]
        second = candidates[1].raw_score if len(candidates) > 1 else 0.0
        margin = float(np.clip((best.raw_score - second) / 0.2, 0.0, 1.0))
        similarity = round(
            100.0
            * float(
                np.clip(
                    0.4 * best.geometry + 0.5 * best.appearance + 0.1 * margin,
                    0.0,
                    1.0,
                )
            ),
            1,
        )
        return MatchResult(
            best.record,
            similarity,
            "sift",
            best.inlier_count,
            best.inlier_ratio,
            best.appearance,
        )

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
        if len(self.index.descriptors):
            with self._flann_lock:
                matches_by_descriptor = self.index.global_matcher.knnMatch(descriptors, k=5)

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
        raise _NoMatchEvidenceError("No primary geometric match evidence")
