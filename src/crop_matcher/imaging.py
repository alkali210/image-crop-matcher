from io import BytesIO
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    pass


class ImageTooLargeError(ValueError):
    pass


def decode_image_bytes(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as header:
                width, height = header.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("Decoded image exceeds the pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageDecodeError("The uploaded file is not a supported image") from exc

    if width <= 0 or height <= 0:
        raise ImageDecodeError("The uploaded file is not a supported image")
    if width * height > max_pixels:
        raise ImageTooLargeError("Decoded image exceeds the pixel limit")

    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageDecodeError("The uploaded file is not a supported image") from exc
    if image is None:
        raise ImageDecodeError("The uploaded file is not a supported image")
    if image.shape[0] * image.shape[1] > max_pixels:
        raise ImageTooLargeError("Decoded image exceeds the pixel limit")
    return image


def read_image(path: Path, max_pixels: int) -> np.ndarray:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageDecodeError(f"Cannot read image: {path.name}") from exc
    return decode_image_bytes(data, max_pixels)


def resize_to_max(image: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image.copy(), scale
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    gx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Correlation inputs must have identical shapes")
    a = left.astype(np.float32).ravel()
    b = right.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def perceptual_hash(gray: np.ndarray) -> np.uint64:
    resized = cv2.resize(to_gray(gray), (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8].ravel()
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    value = 0
    for index, bit in enumerate(bits):
        value |= int(bit) << index
    return np.uint64(value)
