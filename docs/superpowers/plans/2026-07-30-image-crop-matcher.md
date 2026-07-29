# Traditional CV Image Crop Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI application that identifies the source image for a resized color or grayscale crop using traditional computer vision and presents the result in a compact WebUI.

**Architecture:** A deterministic catalog maps safe image IDs to trusted files under `songs/`. A cached SIFT/FLANN index performs global retrieval, candidate-local RANSAC affine verification and normalized appearance comparison produce the primary ranking, and a DCT perceptual-hash tile index supplies a low-feature fallback. FastAPI owns startup state, upload validation, safe image delivery, and static UI hosting.

**Tech Stack:** Python 3.11+, OpenCV 4.x, NumPy 2.x, FastAPI, Uvicorn, python-multipart, pytest, HTTPX, native HTML/CSS/JavaScript.

## Global Constraints

- Use traditional computer vision only; do not add neural-network models, embeddings, OCR, or remote recognition services.
- Index `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files under `songs/`, case-insensitively, excluding stems ending in `_256`.
- Support square crops resized to at least 64 pixels per side and optional grayscale conversion; rotation, mirroring, borders, overlays, and perspective distortion are out of scope.
- Always return the highest-ranked source because queries are expected to belong to the gallery.
- Return similarity in the inclusive range 0-100 as ranking confidence, not probability.
- Reject uploads larger than 10 MiB or decoded images larger than 25 megapixels.
- Keep all processing local and serve native HTML/CSS/JavaScript without Node, CDN, remote fonts, or icon packages.
- Preserve the confirmed compact upload page and thumbnail-plus-information result list; do not restore decorative hero copy or match-evidence prose.
- Target at least 95% Top-1 accuracy on the deterministic benchmark and warm-query P50 below one second on the development machine.

---

## File Structure

```text
pyproject.toml                         Project metadata, runtime/dev dependencies, pytest and Ruff settings
.gitignore                             Existing gallery ignores plus Python, cache, and benchmark artifacts
README.md                              Setup, run, index lifecycle, API, benchmark, and limitations
src/crop_matcher/__init__.py           Package marker and version
src/crop_matcher/config.py             Immutable paths, upload limits, feature and fallback settings
src/crop_matcher/imaging.py            Unicode-safe decode, resize, grayscale, gradients, NCC, and pHash
src/crop_matcher/catalog.py            Source scan, manifest, opaque IDs, and trusted path lookup
src/crop_matcher/feature_index.py      SIFT extraction, FLANN construction, pHash tiles, NPZ cache
src/crop_matcher/matcher.py            Retrieval, RANSAC verification, fallback, ranking, similarity
src/crop_matcher/schemas.py            Pydantic API response models and error body
src/crop_matcher/main.py               FastAPI factory, background index build, endpoints, static hosting
src/crop_matcher/static/index.html     Accessible upload and compact result-list structure
src/crop_matcher/static/styles.css     Confirmed responsive dark-gallery visual design
src/crop_matcher/static/app.js         Status polling, drag/drop upload, API calls, result rendering
benchmarks/benchmark.py                Deterministic crop generation, accuracy and latency reporting
benchmarks/__init__.py                 Makes benchmark helpers importable from tests
tests/test_imaging.py                  Decode limits, resize, NCC, gradients, and pHash tests
tests/test_catalog.py                  Scan, exclusion, ordering, IDs, manifest, and path safety tests
tests/test_feature_index.py            SIFT index, tile index, cache round-trip and invalidation tests
tests/test_matcher.py                  Color, grayscale, low-feature, ranking, and score tests
tests/test_api.py                      Status, upload, response, errors, image serving, and static UI tests
```

The package stays split by responsibility. `matcher.py` consumes catalog records and a completed `FeatureIndex`; it never scans paths or writes cache files. `main.py` translates HTTP requests to those domain interfaces and contains no CV scoring logic.

### Task 1: Project Foundation and Image Primitives

**Files:**
- Create: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/crop_matcher/__init__.py`
- Create: `src/crop_matcher/config.py`
- Create: `src/crop_matcher/imaging.py`
- Create: `tests/test_imaging.py`

**Interfaces:**
- Produces: `Settings`, `decode_image_bytes(data, max_pixels)`, `read_image(path, max_pixels)`, `resize_to_max(image, max_edge)`, `to_gray(image)`, `gradient_magnitude(gray)`, `normalized_correlation(left, right)`, and `perceptual_hash(gray)`.
- Consumes: no project interfaces.

- [ ] **Step 1: Add the failing image primitive tests**

```python
# tests/test_imaging.py
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
```

- [ ] **Step 2: Add project metadata and run the tests to verify collection fails**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "crop-matcher"
version = "0.1.0"
description = "Traditional CV source-image matcher for resized crops"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.116,<1",
  "numpy>=2,<3",
  "opencv-python-headless>=4.10,<5",
  "python-multipart>=0.0.20,<1",
  "uvicorn[standard]>=0.35,<1",
]

[project.optional-dependencies]
dev = [
  "httpx>=0.28,<1",
  "pytest>=8.4,<9",
  "pytest-cov>=6.2,<7",
  "ruff>=0.12,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["src/crop_matcher"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Create `src/crop_matcher/__init__.py` at the same time so the editable package can be installed:

```python
__version__ = "0.1.0"
```

Run: `python -m pip install -e ".[dev]" && pytest tests/test_imaging.py -v`

Expected: test collection fails with `ModuleNotFoundError: No module named 'crop_matcher.imaging'`.

- [ ] **Step 3: Implement settings and image primitives**

```python
# src/crop_matcher/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    gallery_dir: Path = Path("songs")
    cache_dir: Path = Path(".cache")
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    working_max_edge: int = 512
    query_feature_min_edge: int = 256
    sift_features: int = 1_000
    sift_contrast_threshold: float = 0.02
    candidate_count: int = 10
    tile_sizes: tuple[int, ...] = (64, 96, 128, 192, 256)
```

```python
# src/crop_matcher/imaging.py
from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


class ImageTooLargeError(ValueError):
    pass


def decode_image_bytes(data: bytes, max_pixels: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
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
```

Append these lines to the existing `.gitignore` without removing `undefined/`, `songs/`, or `.agents/`:

```gitignore
.cache/
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
*.egg-info/
benchmark-failures/
```

- [ ] **Step 4: Run focused tests and lint**

Run: `pytest tests/test_imaging.py -v && ruff check src tests`

Expected: all image tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the foundation**

```bash
git add pyproject.toml .gitignore src/crop_matcher tests/test_imaging.py
git commit -m "build: add image processing foundation"
```

### Task 2: Deterministic and Safe Image Catalog

**Files:**
- Create: `src/crop_matcher/catalog.py`
- Create: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `read_image(path, max_pixels)` from Task 1 for validation.
- Produces: immutable `ImageRecord`, `CatalogManifestEntry`, and `ImageCatalog.scan(root, max_pixels)`, plus `catalog.records`, `catalog.manifest`, `catalog.get(image_id)`.

- [ ] **Step 1: Write failing catalog behavior tests**

```python
# tests/test_catalog.py
from pathlib import Path

import cv2
import numpy as np
import pytest

from crop_matcher.catalog import ImageCatalog


def write_image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", np.full((20, 20, 3), value, np.uint8))
    assert ok
    path.write_bytes(payload.tobytes())


def test_scan_is_sorted_filters_thumbnails_and_skips_corrupt_files(tmp_path: Path) -> None:
    write_image(tmp_path / "z-song" / "cover.PNG", 20)
    write_image(tmp_path / "a-song" / "base.jpg", 30)
    write_image(tmp_path / "a-song" / "base_256.jpg", 40)
    (tmp_path / "bad.jpg").write_bytes(b"broken")
    (tmp_path / "audio.wav").write_bytes(b"audio")

    catalog = ImageCatalog.scan(tmp_path, max_pixels=10_000)

    assert [record.relative_path.as_posix() for record in catalog.records] == [
        "a-song/base.jpg",
        "z-song/cover.PNG",
    ]
    assert catalog.records[0].parent_name == "a-song"
    assert catalog.get(catalog.records[0].image_id) == catalog.records[0]


def test_ids_are_stable_and_unknown_ids_do_not_resolve(tmp_path: Path) -> None:
    write_image(tmp_path / "song" / "base.jpg")
    first = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    second = ImageCatalog.scan(tmp_path, max_pixels=10_000)
    assert first.records[0].image_id == second.records[0].image_id
    with pytest.raises(KeyError):
        first.get("../../outside")
```

- [ ] **Step 2: Run tests to verify the missing catalog fails**

Run: `pytest tests/test_catalog.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'crop_matcher.catalog'`.

- [ ] **Step 3: Implement the catalog and manifest**

```python
# src/crop_matcher/catalog.py
from dataclasses import dataclass
from hashlib import blake2s
from pathlib import Path

from crop_matcher.imaging import ImageDecodeError, ImageTooLargeError, read_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class CatalogManifestEntry:
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    path: Path
    relative_path: Path
    parent_name: str
    filename: str
    width: int
    height: int


class ImageCatalog:
    def __init__(
        self,
        root: Path,
        records: tuple[ImageRecord, ...],
        manifest: tuple[CatalogManifestEntry, ...],
    ) -> None:
        self.root = root
        self.records = records
        self.manifest = manifest
        self._by_id = {record.image_id: record for record in records}

    @classmethod
    def scan(cls, root: Path, max_pixels: int) -> "ImageCatalog":
        root = root.resolve()
        records: list[ImageRecord] = []
        manifest: list[CatalogManifestEntry] = []
        paths = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not path.stem.lower().endswith("_256")
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        for path in paths:
            relative = path.relative_to(root)
            try:
                image = read_image(path, max_pixels)
            except (ImageDecodeError, ImageTooLargeError):
                continue
            stat = path.stat()
            normalized = relative.as_posix()
            records.append(
                ImageRecord(
                    image_id=blake2s(normalized.encode("utf-8"), digest_size=12).hexdigest(),
                    path=path,
                    relative_path=relative,
                    parent_name=relative.parent.name,
                    filename=relative.name,
                    width=image.shape[1],
                    height=image.shape[0],
                )
            )
            manifest.append(CatalogManifestEntry(normalized, stat.st_size, stat.st_mtime_ns))
        return cls(root, tuple(records), tuple(manifest))

    def get(self, image_id: str) -> ImageRecord:
        return self._by_id[image_id]
```

- [ ] **Step 4: Run catalog and image tests**

Run: `pytest tests/test_catalog.py tests/test_imaging.py -v && ruff check src tests`

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the catalog**

```bash
git add src/crop_matcher/catalog.py tests/test_catalog.py
git commit -m "feat: add deterministic image catalog"
```

### Task 3: Cached SIFT and FLANN Feature Index

**Files:**
- Create: `src/crop_matcher/feature_index.py`
- Create: `tests/test_feature_index.py`

**Interfaces:**
- Consumes: `Settings`, `ImageCatalog`, `ImageRecord`, `read_image`, `resize_to_max`, and `to_gray`.
- Produces: `ImageFeatures`, `TileFeatures`, `FeatureIndex`, and `FeatureIndex.load_or_build(catalog, settings)`; `FeatureIndex.global_matcher` is trained and ready for locked query access.

- [ ] **Step 1: Write failing index and cache tests**

```python
# tests/test_feature_index.py
from pathlib import Path

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex


def write_textured_image(path: Path, offset: int) -> None:
    image = np.zeros((160, 160, 3), np.uint8)
    cv2.putText(image, f"MATCH-{offset}", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.circle(image, (80 + offset, 110), 24, (80, 180, 240), 3)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.tobytes())


def test_builds_global_descriptors_and_round_trips_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 10)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.descriptors.dtype == np.float32
    assert built.descriptors.shape[0] > 0
    assert set(built.by_image) == {record.image_id for record in catalog.records}
    assert loaded.loaded_from_cache is True
    assert np.array_equal(loaded.descriptors, built.descriptors)


def test_manifest_change_invalidates_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    path = gallery / "one" / "base.jpg"
    write_textured_image(path, 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    first_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(first_catalog, settings)
    write_textured_image(path, 20)
    second_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    rebuilt = FeatureIndex.load_or_build(second_catalog, settings)
    assert rebuilt.loaded_from_cache is False
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `pytest tests/test_feature_index.py -v`

Expected: collection fails because `crop_matcher.feature_index` does not exist.

- [ ] **Step 3: Implement SIFT extraction, flattened storage, safe NPZ cache, and FLANN training**

Implement these exact public data shapes in `src/crop_matcher/feature_index.py`:

```python
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.imaging import read_image, resize_to_max, to_gray


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    working_width: int
    working_height: int
    working_scale: float


@dataclass(frozen=True, slots=True)
class TileFeatures:
    hashes: np.ndarray
    image_indices: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    sizes: np.ndarray


class FeatureIndex:
    def __init__(
        self,
        image_ids: tuple[str, ...],
        by_image: dict[str, ImageFeatures],
        descriptors: np.ndarray,
        descriptor_image_indices: np.ndarray,
        tiles: TileFeatures,
        loaded_from_cache: bool,
    ) -> None:
        self.image_ids = image_ids
        self.by_image = by_image
        self.descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        self.descriptor_image_indices = descriptor_image_indices.astype(np.int32, copy=False)
        self.tiles = tiles
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
        manifest = [asdict(entry) for entry in catalog.manifest]
        if manifest_path.exists() and index_path.exists():
            if json.loads(manifest_path.read_text("utf-8")) == manifest:
                return cls._load(index_path)
        index = cls._build(catalog, settings)
        index._save(index_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
        return index
```

Complete `_build` with `cv2.SIFT_create(nfeatures=settings.sift_features, contrastThreshold=settings.sift_contrast_threshold)`. Store queryable point coordinates as contiguous `float32` arrays with shape `(N, 2)`, descriptors as `(N, 128)` `float32`, and an `int32` global descriptor-to-image array. Images with no SIFT features must still have an `ImageFeatures` entry containing empty arrays.

Complete `_save` and `_load` with `np.savez`/`np.load(..., allow_pickle=False)`. Flatten variable-length point and descriptor arrays and persist `int64` offsets; store image IDs as fixed-width Unicode arrays. Write to a temporary file in `cache_dir` and replace `features.npz` only after a successful save. Build the initial `TileFeatures` with correctly typed empty arrays; Task 5 fills it.

- [ ] **Step 4: Run cache tests, then inspect cache safety**

Run: `pytest tests/test_feature_index.py -v && ruff check src tests`

Expected: tests pass; `.cache/features.npz` loads with `allow_pickle=False`; no object-dtype array is present.

- [ ] **Step 5: Commit the primary index**

```bash
git add src/crop_matcher/feature_index.py tests/test_feature_index.py
git commit -m "feat: add cached SIFT feature index"
```

### Task 4: Primary Geometric Matcher and Similarity

**Files:**
- Create: `src/crop_matcher/matcher.py`
- Create: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `ImageCatalog`, `FeatureIndex`, `Settings`, and image primitives.
- Produces: `MatchResult` and `ImageMatcher.match(query_bgr) -> MatchResult`; the result includes `record`, `similarity`, `method`, and candidate diagnostics used only by tests/benchmarking.

- [ ] **Step 1: Write failing end-to-end primary matcher tests**

```python
# tests/test_matcher.py
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
    cv2.putText(image, f"ART-{seed}", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return image


def write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path.write_bytes(payload.tobytes())


@pytest.mark.parametrize("grayscale", [False, True])
def test_matches_resized_crop_to_source(tmp_path: Path, grayscale: bool) -> None:
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

    result = matcher.match(query)

    assert result.record.parent_name == "song-1"
    assert result.method == "sift"
    assert 0.0 <= result.similarity <= 100.0
```

- [ ] **Step 2: Run tests to establish the missing matcher failure**

Run: `pytest tests/test_matcher.py -v`

Expected: collection fails because `crop_matcher.matcher` does not exist.

- [ ] **Step 3: Implement retrieval, affine verification, appearance scoring, and ranking**

Create these domain types and methods in `src/crop_matcher/matcher.py`:

```python
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import gradient_magnitude, normalized_correlation, read_image, resize_to_max, to_gray


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
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
        if descriptors is None or len(descriptors) < 4:
            return self._fallback(query_gray)
        candidate_ids = self._retrieve(descriptors)
        candidates = [
            score
            for image_id in candidate_ids
            if (score := self._verify(image_id, query_gray, points, descriptors, extraction_scale))
            is not None
        ]
        if not candidates:
            return self._fallback(query_gray)
        candidates.sort(key=lambda item: item.raw_score, reverse=True)
        best = candidates[0]
        second = candidates[1].raw_score if len(candidates) > 1 else 0.0
        margin = float(np.clip((best.raw_score - second) / 0.2, 0.0, 1.0))
        similarity = round(100.0 * np.clip(0.4 * best.geometry + 0.5 * best.appearance + 0.1 * margin, 0.0, 1.0), 1)
        return MatchResult(best.record, similarity, "sift", best.inlier_count, best.inlier_ratio, best.appearance)
```

Implement `_feature_query` by upscaling the shortest query edge to 256 with cubic interpolation and returning its scale. Implement `_retrieve` with `global_matcher.knnMatch(descriptors, k=5)` under `_flann_lock`, Lowe filtering, weighted votes, descriptor-row-to-image lookup, and exactly `settings.candidate_count` distinct IDs.

Implement `_verify` with candidate-local `cv2.BFMatcher(cv2.NORM_L2).knnMatch`, ratio `0.78`, at least four good pairs, and `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC, ransacReprojThreshold=4.0)`. Compose extraction scale into the query-to-candidate affine matrix, reject non-finite or implausible scale and mapped corners, then call `cv2.warpAffine(candidate_gray, query_to_candidate, query_size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)`. Map grayscale and Sobel NCC from `[-1, 1]` to `[0, 1]`; appearance is `0.7 * gray + 0.3 * edge`. Geometry combines inlier ratio, capped inlier count, and reprojection quality.

Define `_fallback` to raise a private `NoMatchEvidenceError` until Task 5 replaces it. Add a temporary test assertion that primary textured fixtures never reach this branch.

- [ ] **Step 4: Run primary matcher tests repeatedly with a fixed fixture**

Run: `pytest tests/test_matcher.py -v && pytest tests/test_imaging.py tests/test_catalog.py tests/test_feature_index.py -v`

Expected: color and grayscale crops both resolve to `song-1`; all earlier tests remain green.

- [ ] **Step 5: Commit the primary matcher**

```bash
git add src/crop_matcher/matcher.py tests/test_matcher.py
git commit -m "feat: match crops with SIFT geometry"
```

### Task 5: Low-Feature Tile Hash Fallback

**Files:**
- Modify: `src/crop_matcher/feature_index.py`
- Modify: `src/crop_matcher/matcher.py`
- Modify: `tests/test_feature_index.py`
- Modify: `tests/test_matcher.py`

**Interfaces:**
- Consumes: Task 3 `TileFeatures` fields and Task 4 `ImageMatcher._fallback` hook.
- Produces: populated 64-bit pHash tiles and a fallback `MatchResult(method="phash")` capped at 89.9 similarity.

- [ ] **Step 1: Add failing tile construction and fallback tests**

```python
def test_feature_index_builds_typed_hash_tiles(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    index = FeatureIndex.load_or_build(catalog, settings)
    assert index.tiles.hashes.dtype == np.uint64
    assert len(index.tiles.hashes) > 0
    assert len(index.tiles.hashes) == len(index.tiles.image_indices)
```

```python
def test_low_feature_crop_uses_fallback(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    first = np.zeros((320, 320, 3), np.uint8)
    second = np.zeros((320, 320, 3), np.uint8)
    cv2.rectangle(first, (40, 40), (280, 280), (80, 80, 80), -1)
    cv2.rectangle(first, (110, 110), (210, 210), (180, 180, 180), -1)
    cv2.rectangle(second, (40, 40), (280, 280), (80, 80, 80), -1)
    cv2.circle(second, (160, 160), 50, (180, 180, 180), -1)
    write_jpg(gallery / "square" / "base.jpg", first)
    write_jpg(gallery / "circle" / "base.jpg", second)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    matcher = ImageMatcher(catalog, FeatureIndex.load_or_build(catalog, settings), settings)
    query = cv2.resize(first[80:240, 80:240], (64, 64))
    result = matcher.match(query)
    assert result.record.parent_name == "square"
    assert result.method == "phash"
    assert result.similarity <= 89.9
```

- [ ] **Step 2: Run the new tests and verify they fail for empty tiles/fallback**

Run: `pytest tests/test_feature_index.py::test_feature_index_builds_typed_hash_tiles tests/test_matcher.py::test_low_feature_crop_uses_fallback -v`

Expected: the tile assertion fails and the matcher raises `NoMatchEvidenceError`.

- [ ] **Step 3: Populate tile features and implement fallback verification**

In `FeatureIndex._build`, iterate each working grayscale image with tile sizes `(64, 96, 128, 192, 256)`, skip sizes larger than either image dimension, and use `stride = size // 2`. Always add the final right and bottom aligned positions so edge regions are indexed. Store pHash, image index, x, y, and size in the typed `TileFeatures` arrays; include these arrays in `_save` and `_load`.

Replace `_fallback` with this behavior:

```python
def _fallback(self, query_gray: np.ndarray) -> MatchResult:
    query_hash = perceptual_hash(query_gray)
    xor = np.bitwise_xor(self.index.tiles.hashes, query_hash)
    distances = np.fromiter(
        (int(value).bit_count() for value in xor),
        dtype=np.uint8,
        count=len(xor),
    )
    order = np.argsort(distances, kind="stable")
    candidate_indices: list[int] = []
    for tile_index in order:
        image_index = int(self.index.tiles.image_indices[tile_index])
        if image_index not in candidate_indices:
            candidate_indices.append(image_index)
        if len(candidate_indices) == self.settings.candidate_count:
            break
    scored = [self._template_score(index, query_gray) for index in candidate_indices]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_index, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = float(np.clip((best_score - second_score) / 0.2, 0.0, 1.0))
    similarity = round(min(89.9, 100.0 * (0.85 * best_score + 0.15 * margin)), 1)
    record = self.catalog.get(self.index.image_ids[best_index])
    return MatchResult(record, similarity, "phash", 0, 0.0, best_score)
```

Implement `_template_score` by loading and downscaling the candidate, comparing query grayscale and Sobel gradients through `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` over candidate pyramids derived from the indexed tile sizes, and returning the best mapped `[0, 1]` score. If a query is larger than a candidate level, skip that level. Raise `NoMatchEvidenceError` only when the catalog or tile index is empty, which startup treats as index failure.

- [ ] **Step 4: Run fallback, primary, and cache round-trip tests**

Run: `pytest tests/test_feature_index.py tests/test_matcher.py -v && ruff check src tests`

Expected: both primary methods and the fallback pass; cache reload preserves non-empty tile arrays.

- [ ] **Step 5: Commit the fallback**

```bash
git add src/crop_matcher/feature_index.py src/crop_matcher/matcher.py tests/test_feature_index.py tests/test_matcher.py
git commit -m "feat: add low-feature hash fallback"
```

### Task 6: FastAPI Lifecycle, Validation, and Safe Image API

**Files:**
- Create: `src/crop_matcher/schemas.py`
- Create: `src/crop_matcher/main.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `ImageCatalog`, `FeatureIndex.load_or_build`, `ImageMatcher.match`, and `MatchResult`.
- Produces: `create_app(settings: Settings | None = None) -> FastAPI`, module-level `app`, and the three confirmed API routes.

- [ ] **Step 1: Write failing API contract and validation tests**

```python
# tests/test_api.py
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from crop_matcher.config import Settings
from crop_matcher.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    gallery = tmp_path / "songs"
    image = np.zeros((220, 220, 3), np.uint8)
    cv2.putText(image, "API", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    path = gallery / "api-song" / "base.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload.tobytes())
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    return TestClient(create_app(settings))


def test_status_match_and_safe_image_delivery(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["state"] in {"building", "ready"}
        for _ in range(100):
            if client.get("/api/status").json()["state"] == "ready":
                break
            time.sleep(0.01)
        assert client.get("/api/status").json()["state"] == "ready"
        source = cv2.imread(str(tmp_path / "songs" / "api-song" / "base.jpg"))
        query = cv2.resize(source[30:190, 30:190], (90, 90))
        ok, payload = cv2.imencode(".png", query)
        assert ok
        response = client.post("/api/match", files={"file": ("query.png", payload.tobytes(), "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["matches"][0]["parent_name"] == "api-song"
        image_response = client.get(body["matches"][0]["image_url"])
        assert image_response.status_code == 200
        assert image_response.headers["content-type"].startswith("image/")
        assert client.get("/api/images/../../outside").status_code in {404, 422}


def test_rejects_invalid_and_oversized_uploads(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        invalid = client.post("/api/match", files={"file": ("bad.jpg", b"bad", "image/jpeg")})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_image"
        oversized = client.post(
            "/api/match",
            files={"file": ("large.bin", b"x" * (10 * 1024 * 1024 + 1), "application/octet-stream")},
        )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "file_too_large"
```

- [ ] **Step 2: Run API tests to verify missing application failure**

Run: `pytest tests/test_api.py -v`

Expected: collection fails because `crop_matcher.main` does not exist.

- [ ] **Step 3: Implement schemas, background startup, endpoints, and structured errors**

```python
# src/crop_matcher/schemas.py
from typing import Literal

from pydantic import BaseModel


class StatusResponse(BaseModel):
    state: Literal["building", "ready", "error"]
    indexed_images: int
    build_time_ms: int | None
    error: str | None


class QueryInfo(BaseModel):
    width: int
    height: int


class MatchItem(BaseModel):
    image_id: str
    parent_name: str
    filename: str
    width: int
    height: int
    similarity: float
    image_url: str


class MatchResponse(BaseModel):
    query: QueryInfo
    elapsed_ms: int
    matches: list[MatchItem]
```

In `main.py`, define an `AppServices` dataclass with state, error, build time, catalog, feature index, and matcher. `create_app` creates this state and uses a lifespan context manager to schedule one `asyncio.create_task(asyncio.to_thread(build_services))`; shutdown awaits or cancels it safely. `build_services` scans the catalog, rejects an empty catalog, loads/builds the feature index, constructs `ImageMatcher`, and atomically changes state to `ready` or `error`.

Implement `GET /api/status` from `AppServices`. Implement `POST /api/match` by reading `max_upload_bytes + 1`, returning `413` before decode when oversized, decoding through `decode_image_bytes`, returning `503` unless ready, and running `matcher.match` through `asyncio.to_thread`. Serialize exactly one match item and measured `elapsed_ms`. Implement `GET /api/images/{image_id}` with catalog lookup and `FileResponse`; never join a client-provided path. Add exception handlers returning:

```json
{"error": {"code": "invalid_image", "message": "The uploaded file is not a supported image"}}
```

Mount the static directory at `/static` and return `index.html` from `/`. Export `app = create_app()`.

- [ ] **Step 4: Run API and full Python tests**

Run: `pytest -v && ruff check src tests`

Expected: API status reaches `ready`, valid upload returns the expected schema, invalid upload returns structured `400`, and all tests pass.

- [ ] **Step 5: Commit the API**

```bash
git add src/crop_matcher/schemas.py src/crop_matcher/main.py tests/test_api.py
git commit -m "feat: expose matcher through FastAPI"
```

### Task 7: Confirmed Responsive WebUI

**Files:**
- Create: `src/crop_matcher/static/index.html`
- Create: `src/crop_matcher/static/styles.css`
- Create: `src/crop_matcher/static/app.js`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `GET /api/status`, `POST /api/match`, and match item `image_url` from Task 6.
- Produces: accessible upload UI with index state, loading/error states, compact result rows, original-image links, and re-upload flow.

- [ ] **Step 1: Add failing static shell assertions**

```python
def test_root_serves_functional_static_shell(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="drop-zone"' in response.text
        assert 'id="result-list"' in response.text
        assert "Find the frame" not in response.text
        assert "几何内点稳定" not in response.text
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
```

- [ ] **Step 2: Run the shell test and verify missing files fail**

Run: `pytest tests/test_api.py::test_root_serves_functional_static_shell -v`

Expected: `GET /` fails or references missing static files.

- [ ] **Step 3: Implement the approved upload and list UI**

Create semantic `index.html` with a compact header, `aria-live` index state, centered drop zone, hidden file input, loading/error region, hidden query summary, result heading, empty ordered result list, and re-upload button. Link only local `/static/styles.css` and `/static/app.js`.

Implement `app.js` with one explicit state object and these functions:

```javascript
const state = { selectedFile: null, queryUrl: null, statusTimer: null };

async function pollStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  renderStatus(status);
  if (status.state === "building") {
    state.statusTimer = window.setTimeout(pollStatus, 750);
  }
}

async function submitFile(file) {
  setBusy(true);
  clearError();
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("/api/match", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || "匹配失败，请重试");
    renderQuery(file);
    renderMatches(body.matches);
    showResults();
  } catch (error) {
    showError(error instanceof Error ? error.message : "匹配失败，请重试");
  } finally {
    setBusy(false);
  }
}

function renderMatches(matches) {
  const list = document.querySelector("#result-list");
  list.replaceChildren(...matches.map((match, index) => createMatchRow(match, index + 1)));
}
```

Complete `renderStatus`, `setBusy`, `clearError`, `showError`, `renderQuery`, `showResults`, and `createMatchRow` without `innerHTML` for API-derived text. Wire click, keyboard, drag-enter/leave/drop, change, and re-upload events. Revoke old object URLs before replacing query previews.

Implement `styles.css` from the approved revision: charcoal background, warm-white typography, muted green ready indicator, restrained coral accent, one centered upload panel, and compact three-column result rows (`150px minmax(0,1fr) 150px`). At `700px`, collapse the score to a full-width row; keep every interactive target at least 44px. Add visible focus styles and a `prefers-reduced-motion: reduce` rule.

- [ ] **Step 4: Verify API tests and browser behavior at two viewports**

Run: `pytest tests/test_api.py -v && ruff check src tests`

Expected: all static and API tests pass.

Run the app: `uvicorn crop_matcher.main:app --reload`

Browser checks:

- Desktop `1440x900`: index state and count appear above one upload box; one result occupies a compact list row rather than the full screen.
- Mobile `390x844`: upload remains usable, result metadata does not overflow, score and original link wrap below the thumbnail/info row.
- Upload a valid crop: loading state disables duplicate submission, then result shows thumbnail, parent directory, filename, dimensions, rank, similarity, and original link.
- Upload invalid bytes: concise error is visible and re-upload remains available.

- [ ] **Step 5: Commit the WebUI**

```bash
git add src/crop_matcher/static tests/test_api.py
git commit -m "feat: add compact matcher web interface"
```

### Task 8: Deterministic Benchmark, Documentation, and Final Verification

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/benchmark.py`
- Create: `README.md`
- Modify: `tests/test_matcher.py`

**Interfaces:**
- Consumes: all production interfaces and the real `songs/` gallery.
- Produces: reproducible accuracy/latency report, saved failure metadata, and complete operator instructions.

- [ ] **Step 1: Add a deterministic benchmark smoke test**

Add a test that imports benchmark generation without running the full gallery:

```python
def test_benchmark_queries_are_deterministic() -> None:
    from benchmarks.benchmark import crop_specs

    first = list(crop_specs(seed=20260730, image_count=3, samples_per_image=2))
    second = list(crop_specs(seed=20260730, image_count=3, samples_per_image=2))
    assert first == second
    assert {spec.output_size for spec in first}.issubset({64, 90, 128, 192})
    assert all(0.10 <= spec.crop_fraction <= 0.40 for spec in first)
```

- [ ] **Step 2: Run the smoke test and verify the benchmark module is missing**

Run: `pytest tests/test_matcher.py::test_benchmark_queries_are_deterministic -v`

Expected: import fails because `benchmarks/benchmark.py` does not exist.

- [ ] **Step 3: Implement benchmark CLI and operator documentation**

Create an empty `benchmarks/__init__.py`. Create `benchmarks/benchmark.py` with a frozen `CropSpec` dataclass, fixed seed default `20260730`, crop fractions sampled uniformly from `[0.10, 0.40]`, output sizes selected from `(64, 90, 128, 192)`, and an alternating color/grayscale flag. The CLI accepts `--gallery`, `--samples-per-image`, `--max-images`, `--seed`, and `--failure-dir`. It builds one catalog/index/matcher, performs one warm-up, measures each query with `time.perf_counter_ns`, and prints:

```text
images=<count> queries=<count>
top1=<correct>/<total> accuracy=<percent>%
method_sift=<count> method_phash=<count>
latency_ms_p50=<value> latency_ms_p95=<value>
```

For each incorrect result, write one JSON file containing source ID, predicted ID, crop x/y/side, output size, grayscale flag, similarity, and latency. Exit with status 1 when Top-1 is below 95%; report latency without making hardware-dependent P50 a hard process failure.

Write `README.md` with:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000
python benchmarks/benchmark.py --gallery songs --samples-per-image 4
pytest -v
```

Document first-build versus cached startup, restart-required gallery updates, supported transforms/formats, 10 MiB and 25 MP limits, the three API routes, similarity semantics, the guaranteed-best-result behavior, and ambiguity for identical/featureless regions.

- [ ] **Step 4: Run complete verification and inspect failures before tuning**

Run: `ruff check .`

Expected: no lint errors.

Run: `pytest -v --cov=crop_matcher --cov-report=term-missing`

Expected: all tests pass; untested lines are reviewed rather than hidden by exclusions.

Run: `python benchmarks/benchmark.py --gallery songs --samples-per-image 4 --seed 20260730`

Expected: Top-1 is at least 95%; report includes SIFT/fallback counts and P50/P95. If accuracy misses the target, use saved failure metadata to tune only the constants already defined in `Settings` and scoring functions, add a regression fixture for each failure class, and repeat all three commands.

Run: `uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000`

Expected: startup reaches `ready`, browser upload returns the correct source, and the app has no console errors at desktop and mobile viewport sizes.

- [ ] **Step 5: Commit benchmark and documentation**

```bash
git add benchmarks/__init__.py benchmarks/benchmark.py tests/test_matcher.py README.md
git commit -m "test: add matcher benchmark and run guide"
```

## Final Review Checklist

- [ ] `git status --short` contains no generated cache, gallery, Python cache, benchmark failure, or unintended file.
- [ ] `git diff --check` reports no whitespace errors.
- [ ] Every source image rule and API field matches the approved design specification.
- [ ] No dependency, source file, or frontend asset contains a neural model or remote service call.
- [ ] A cold start builds the index and a second start loads cached arrays with `allow_pickle=False`.
- [ ] The benchmark records deterministic failures and reaches the accepted Top-1 target before completion is claimed.
- [ ] Browser verification covers upload, result, original-image link, invalid input, re-upload, desktop, and mobile.
