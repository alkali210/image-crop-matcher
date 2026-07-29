from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from crop_matcher import matcher as matcher_module
from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex, ImageFeatures
from crop_matcher.matcher import CandidateScore, ImageMatcher


def make_art(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 70, (320, 320, 3), dtype=np.uint8)
    for index in range(18):
        center = tuple(int(value) for value in rng.integers(20, 300, 2))
        color = tuple(int(value) for value in rng.integers(100, 256, 3))
        cv2.circle(image, center, 5 + index, color, 2)
    cv2.putText(
        image,
        f"ART-{seed}",
        (70, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
    )
    return image


def write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path.write_bytes(payload.tobytes())


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(payload.tobytes())


class FakeGlobalMatcher:
    def __init__(self, matches: list[list[SimpleNamespace]]) -> None:
        self.matches = matches
        self.requested_k: int | None = None

    def knnMatch(self, _descriptors: np.ndarray, k: int) -> list[list[SimpleNamespace]]:
        self.requested_k = k
        return self.matches


class RecordingGlobalMatcher:
    def __init__(self, descriptors: np.ndarray) -> None:
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 64},
        )
        self.matcher.add([descriptors])
        self.matcher.train()
        self.requested_k: int | None = None

    def knnMatch(self, descriptors: np.ndarray, k: int) -> list[list[cv2.DMatch]]:
        self.requested_k = k
        return self.matcher.knnMatch(descriptors, k=k)


def neighbor(train_index: int, distance: float) -> SimpleNamespace:
    return SimpleNamespace(trainIdx=train_index, distance=distance)


def retrieval_matcher(
    image_ids: tuple[str, ...],
    owners: list[int],
    matches: list[list[SimpleNamespace]],
    candidate_count: int,
) -> tuple[ImageMatcher, FakeGlobalMatcher]:
    global_matcher = FakeGlobalMatcher(matches)
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.index = SimpleNamespace(
        descriptors=np.zeros((len(owners), 128), np.float32),
        descriptor_image_indices=np.asarray(owners, np.int32),
        image_ids=image_ids,
        global_matcher=global_matcher,
    )
    matcher.settings = Settings(candidate_count=candidate_count)
    matcher._flann_lock = Lock()
    return matcher, global_matcher


@pytest.mark.parametrize("grayscale", [False, True])
def test_matches_resized_crop_to_source(
    tmp_path: Path, grayscale: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    sources = [make_art(seed) for seed in range(3)]
    for seed, image in enumerate(sources):
        write_jpg(gallery / f"song-{seed}" / "base.jpg", image)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    index = FeatureIndex.load_or_build(catalog, settings)
    matcher = ImageMatcher(catalog, index, settings)
    query = cv2.resize(sources[1][90:210, 130:250], (90, 90))
    if grayscale:
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        query = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)

    def fail_fallback(_query_gray: np.ndarray) -> None:
        pytest.fail("textured primary fixture reached fallback")

    monkeypatch.setattr(matcher, "_fallback", fail_fallback)

    result = matcher.match(query)

    assert result.record.parent_name == "song-1"
    assert result.method == "sift"
    assert 0.0 <= result.similarity <= 100.0


def test_retrieve_uses_five_neighbors_without_duplicate_owner_votes() -> None:
    matcher, global_matcher = retrieval_matcher(
        ("image-0", "image-1", "image-2"),
        [0, 0, 1, 2, 2],
        [
            [
                neighbor(0, 10.0),
                neighbor(1, 20.0),
                neighbor(2, 40.0),
                neighbor(3, 400.0),
                neighbor(4, 410.0),
            ],
            [
                neighbor(2, 14.0),
                neighbor(3, 20.0),
                neighbor(4, 21.0),
                neighbor(0, 22.0),
                neighbor(1, 23.0),
            ],
        ],
        candidate_count=2,
    )

    result = matcher._retrieve(np.zeros((2, 128), np.float32))

    assert global_matcher.requested_k == 5
    assert result == ["image-1", "image-0"]


def test_retrieve_ignores_unsafe_owners_and_deterministically_fills() -> None:
    matcher, _ = retrieval_matcher(
        tuple(f"image-{index}" for index in range(5)),
        [2, 99, 1, 3, 4],
        [
            [
                neighbor(0, 10.0),
                neighbor(1, 20.0),
                neighbor(99, 40.0),
                neighbor(2, 80.0),
                neighbor(3, 81.0),
            ]
        ],
        candidate_count=4,
    )

    result = matcher._retrieve(np.zeros((1, 128), np.float32))

    assert result == ["image-2", "image-0", "image-1", "image-3"]
    assert len(result) == len(set(result)) == 4


@pytest.mark.parametrize(
    ("descriptor_count", "expected_k", "expected"),
    [
        (1, None, ["image-0", "image-1", "image-2"]),
        (2, 2, ["image-2", "image-0", "image-1"]),
        (3, 3, ["image-2", "image-3", "image-0"]),
        (4, 4, ["image-2", "image-3", "image-4"]),
    ],
)
def test_retrieve_caps_knn_to_available_descriptor_rows(
    descriptor_count: int, expected_k: int | None, expected: list[str]
) -> None:
    descriptors = np.repeat(
        (10.0 * np.arange(descriptor_count, dtype=np.float32))[:, None],
        128,
        axis=1,
    )
    global_matcher = RecordingGlobalMatcher(descriptors)
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher.index = SimpleNamespace(
        descriptors=descriptors,
        descriptor_image_indices=np.asarray([2, 3, 4, 1][:descriptor_count], np.int32),
        image_ids=tuple(f"image-{index}" for index in range(5)),
        global_matcher=global_matcher,
    )
    matcher.settings = Settings(candidate_count=3)
    matcher._flann_lock = Lock()

    result = matcher._retrieve(descriptors[:1])

    assert global_matcher.requested_k == expected_k
    assert result == expected
    assert len(result) == len(set(result)) == 3


def test_verify_accepts_large_scale_from_downscaled_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = cv2.resize(make_art(7), (64, 64), interpolation=cv2.INTER_AREA)
    source = np.zeros((1600, 1600, 3), np.uint8)
    source[32:1568, 32:1568] = np.repeat(np.repeat(query, 24, axis=0), 24, axis=1)
    gallery = tmp_path / "songs"
    write_png(gallery / "large" / "base.png", source)
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        working_max_edge=400,
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    record = catalog.records[0]
    query_points = np.asarray(
        [[0.0, 0.0], [252.0, 0.0], [252.0, 252.0], [0.0, 252.0]],
        np.float32,
    )
    feature_affine = np.asarray([[1.5, 0.0, 8.0], [0.0, 1.5, 8.0]], np.float64)
    candidate_points = cv2.transform(query_points[None, :, :], feature_affine)[0]
    descriptors = np.zeros((4, 128), np.float32)
    features = ImageFeatures(candidate_points, descriptors, 400, 400, 0.25)
    index = SimpleNamespace(by_image={record.image_id: features})
    matcher = ImageMatcher(catalog, index, settings)

    pairs = [
        [
            SimpleNamespace(queryIdx=index, trainIdx=index, distance=1.0),
            SimpleNamespace(queryIdx=index, trainIdx=(index + 1) % 4, distance=2.0),
        ]
        for index in range(4)
    ]
    local_matcher = SimpleNamespace(knnMatch=lambda *_args, **_kwargs: pairs)
    monkeypatch.setattr(cv2, "BFMatcher", lambda _norm: local_matcher)
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        lambda *_args, **_kwargs: (feature_affine, np.ones((4, 1), np.uint8)),
    )

    score = matcher._verify(
        record.image_id,
        cv2.cvtColor(query, cv2.COLOR_BGR2GRAY),
        query_points,
        descriptors,
        extraction_scale=4.0,
    )

    assert features.working_scale == pytest.approx(0.25)
    assert score is not None
    assert score.appearance > 0.95


def test_mapped_geometry_uses_per_axis_tolerance() -> None:
    valid = np.asarray([[0.0, -4.0], [999.0, -4.0], [999.0, 99.0], [0.0, 99.0]])
    outside_short_axis = np.asarray([[0.0, -6.0], [999.0, -6.0], [999.0, 99.0], [0.0, 99.0]])

    assert ImageMatcher._mapped_geometry_is_valid(valid, 1000, 100)
    assert not ImageMatcher._mapped_geometry_is_valid(outside_short_axis, 1000, 100)


def test_match_ranks_by_raw_score_and_applies_bounded_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best_record = ImageRecord("best", Path("best.png"), Path("best.png"), "", "best.png", 1, 1)
    second_record = ImageRecord(
        "second", Path("second.png"), Path("second.png"), "", "second.png", 1, 1
    )
    scores = {
        "best": CandidateScore(best_record, 0.5, 0.5, 0.45, 8, 0.8),
        "second": CandidateScore(second_record, 0.5, 0.5, 0.44, 8, 0.8),
    }
    matcher = ImageMatcher.__new__(ImageMatcher)
    matcher._sift = SimpleNamespace(
        detectAndCompute=lambda *_args: (
            [SimpleNamespace(pt=(float(index), float(index))) for index in range(4)],
            np.zeros((4, 128), np.float32),
        )
    )
    monkeypatch.setattr(matcher, "_feature_query", lambda query: (query, 1.0))
    monkeypatch.setattr(matcher, "_retrieve", lambda _descriptors: ["second", "best"])
    monkeypatch.setattr(matcher, "_verify", lambda image_id, *_args: scores[image_id])

    result = matcher.match(np.zeros((8, 8), np.uint8))

    assert result.record == best_record
    assert result.similarity == pytest.approx(45.5)
    assert 0.0 <= result.similarity <= 100.0


def test_no_evidence_uses_private_task5_hook() -> None:
    error_type = getattr(matcher_module, "_NoMatchEvidenceError", None)

    assert error_type is not None
    assert not hasattr(matcher_module, "NoMatchEvidenceError")
    with pytest.raises(error_type, match="No primary geometric match evidence"):
        ImageMatcher.__new__(ImageMatcher)._fallback(np.zeros((8, 8), np.uint8))
