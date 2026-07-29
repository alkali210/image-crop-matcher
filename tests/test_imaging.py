from pathlib import Path

import cv2
import numpy as np
import pytest

from crop_matcher.imaging import (
    ImageDecodeError,
    ImageTooLargeError,
    decode_image_bytes,
    normalized_correlation,
    perceptual_hash,
    read_image,
    resize_to_max,
)


def encode(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, payload = cv2.imencode(extension, image)
    assert ok
    return payload.tobytes()


def test_decode_rejects_invalid_and_excessive_pixels() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"not-an-image", max_pixels=100)

    image = np.zeros((11, 10, 3), dtype=np.uint8)
    with pytest.raises(ImageTooLargeError):
        decode_image_bytes(encode(image), max_pixels=100)


def test_read_image_supports_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "图库.png"
    path.write_bytes(encode(np.full((12, 8, 3), 127, dtype=np.uint8)))
    assert read_image(path, max_pixels=1_000).shape == (12, 8, 3)


def test_resize_correlation_and_hash_are_stable() -> None:
    image = np.arange(80 * 40, dtype=np.uint8).reshape(40, 80)
    resized, scale = resize_to_max(image, 20)
    assert resized.shape == (10, 20)
    assert scale == pytest.approx(0.25)
    assert normalized_correlation(image, image.copy()) == pytest.approx(1.0)
    assert perceptual_hash(image) == perceptual_hash(image.copy())
