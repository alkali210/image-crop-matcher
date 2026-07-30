from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

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


def compressed_png(width: int, height: int) -> bytes:
    payload = BytesIO()
    Image.new("1", (width, height)).save(payload, format="PNG")
    return payload.getvalue()


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP", "BMP"])
def test_required_formats_round_trip_through_pillow_header_and_opencv(
    image_format: str,
) -> None:
    source = Image.new("RGB", (13, 9), (23, 101, 207))
    payload = BytesIO()
    source.save(payload, format=image_format)

    decoded = decode_image_bytes(payload.getvalue(), max_pixels=1_000)

    assert decoded.shape == (9, 13, 3)
    assert decoded.dtype == np.uint8


def test_decode_rejects_invalid_and_excessive_pixels() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"not-an-image", max_pixels=100)

    image = np.zeros((11, 10, 3), dtype=np.uint8)
    with pytest.raises(ImageTooLargeError):
        decode_image_bytes(encode(image), max_pixels=100)


def test_decode_rejects_declared_oversized_image_before_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decode(*_args: object) -> None:
        pytest.fail("cv2.imdecode must not receive an oversized image")

    monkeypatch.setattr(cv2, "imdecode", fail_decode)

    with pytest.raises(ImageTooLargeError):
        decode_image_bytes(compressed_png(5_001, 5_000), max_pixels=25_000_000)


def test_decode_wraps_opencv_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_decode(*_args: object) -> None:
        raise cv2.error("OpenCV decode failed")

    monkeypatch.setattr(cv2, "imdecode", fail_decode)

    with pytest.raises(ImageDecodeError, match="not a supported image"):
        decode_image_bytes(encode(np.zeros((8, 8, 3), np.uint8)), max_pixels=100)


@pytest.mark.parametrize(
    "bomb_type",
    [Image.DecompressionBombError, Image.DecompressionBombWarning],
)
def test_decode_maps_pillow_decompression_bombs_to_image_too_large(
    bomb_type: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_header(*_args: object, **_kwargs: object) -> None:
        raise bomb_type("declared dimensions are unsafe")

    def fail_decode(*_args: object) -> None:
        pytest.fail("cv2.imdecode must not run after a Pillow bomb condition")

    monkeypatch.setattr(Image, "open", fail_header)
    monkeypatch.setattr(cv2, "imdecode", fail_decode)

    with pytest.raises(ImageTooLargeError):
        decode_image_bytes(b"image header", max_pixels=25_000_000)


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
