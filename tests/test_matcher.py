from pathlib import Path

import cv2
import numpy as np
import pytest

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.matcher import ImageMatcher


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
