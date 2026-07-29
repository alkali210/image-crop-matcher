# Traditional CV Image Crop Matcher Design

## Summary

Build a local FastAPI application that receives a square screenshot and finds the source image under `songs/`. A query may be a resized crop of the source or a grayscale version of that crop. The implementation must use traditional computer vision only; neural networks and remote services are out of scope.

The application returns the best matching original image, its parent directory name, its file name, and a 0-100 similarity score. The API and result UI use a list shape from the beginning so Top-K results can be added later without redesigning the response.

## Goals

- Index every supported source image under `songs/`, excluding generated `*_256` thumbnails.
- Match color or grayscale crops with a minimum expected side length of 64 pixels.
- Return the best source image on a normal desktop CPU with a warm-query median latency target below one second.
- Reach at least 95% Top-1 accuracy on a deterministic synthetic benchmark generated from the real gallery.
- Serve a small, responsive, dependency-light WebUI directly from FastAPI.
- Keep all processing and files local.

## Non-Goals

- Neural-network embeddings, OCR, or external recognition APIs.
- Rotation, mirroring, perspective distortion, borders, watermarks, or partial UI overlays in query images.
- Live filesystem watching or incremental index updates while the service runs.
- User accounts, query history, batch upload, or manual gallery management.
- Guaranteed disambiguation when multiple source images contain the same region or a query is nearly featureless.

## Confirmed Constraints

- Query transforms are limited to square cropping, resizing, and optional grayscale conversion.
- A query is expected to belong to the gallery, so the service always returns its highest-ranked candidate.
- The gallery is scanned during application startup. Gallery changes take effect after a restart.
- The current gallery contains 1,397 readable images. Excluding 601 `*_256` thumbnails leaves 796 source images.
- Supported source extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`, compared case-insensitively.
- The WebUI uses native HTML, CSS, and JavaScript with no Node build step, CDN, remote font, or icon dependency.

## Architecture

The application has five bounded components:

1. `ImageCatalog` scans the gallery, applies the source-file rules, assigns opaque image IDs, and resolves IDs back to trusted paths.
2. `FeatureIndex` extracts or loads cached SIFT descriptors and perceptual-hash tiles, then builds the in-memory FLANN index.
3. `ImageMatcher` performs candidate retrieval, geometric verification, local appearance verification, fallback retrieval, ranking, and score calculation.
4. The FastAPI layer owns startup state, upload validation, thread-pool execution, response serialization, and safe original-image delivery.
5. The static WebUI polls index status, uploads one query, and renders the returned matches as compact list rows.

FastAPI initializes the catalog and index in its lifespan handler. The status endpoint reports `building` until the index is usable. CPU-bound matching runs in a worker thread and a process-local lock protects the shared FLANN matcher.

## Catalog and Cache

The catalog recursively scans `songs/` in deterministic relative-path order. It ignores non-image files and any image whose stem ends with `_256`. A corrupt source image is logged and skipped without aborting the complete build. If no valid source remains, startup state becomes `error`.

Each image receives an opaque stable ID derived from its normalized relative path. The original path is never accepted from an HTTP client. The image endpoint resolves only IDs already present in the catalog, preventing directory traversal.

The index cache is stored below the application's `.cache/` directory and contains only arrays and JSON metadata, not Python pickle data. Its manifest records each source's relative path, byte size, and nanosecond modification time. Any manifest difference invalidates the entire cache and causes a rebuild on the next startup. Cached descriptors avoid repeated feature extraction; the FLANN tree is rebuilt in memory from those descriptors.

## Primary Matching Pipeline

### Index Build

1. Decode each source image and record its original dimensions.
2. Downscale it without upscaling so its longest edge is at most 512 pixels, recording the scale factor.
3. Convert it to grayscale.
4. Extract up to 1,000 SIFT keypoints and float descriptors with `contrastThreshold=0.02` for artwork with soft local contrast.
5. Store per-image keypoints, descriptor ranges, dimensions, and scales.
6. Concatenate descriptors and build a FLANN KD-Tree with a row-to-image-ID lookup.
7. Build grayscale perceptual-hash tiles at several square window sizes for the low-feature fallback.

### Query Processing

1. Decode and validate the uploaded image.
2. Preserve its original dimensions and grayscale pixels for appearance verification.
3. For SIFT extraction only, upscale queries whose shortest edge is below 256 pixels.
4. Query the global FLANN index with K-nearest-neighbor search and aggregate descriptor votes by source image.
5. Select the strongest 10 distinct source candidates.
6. Match the query descriptors independently against each candidate with a Lowe ratio test.
7. Estimate a scale-and-translation-compatible affine transform with RANSAC and reject degenerate transforms.
8. Transform the query corners into candidate coordinates, sample the corresponding source region, and compare it with the query.

Affine verification is preferred over a free homography because the confirmed transform set does not include perspective distortion. It needs fewer reliable correspondences and is less likely to accept a geometrically implausible match.

### Local Appearance Verification

The candidate region is warped to query dimensions. Verification compares:

- Normalized grayscale cross-correlation.
- Normalized cross-correlation of Sobel gradient magnitude.
- RANSAC inlier count, inlier ratio, and reprojection error.
- The margin between the best and second-best candidates.

The initial similarity formula maps each term to `[0, 1]` and uses 50% local appearance, 40% geometric quality, and 10% candidate margin. The public score is the clamped result multiplied by 100 and rounded to one decimal place. This score is a ranking confidence, not a probability. Formula weights may be calibrated against the committed benchmark fixture, but the API meaning and 0-100 range remain fixed.

## Low-Feature Fallback

The fallback runs only when the query has too few usable SIFT descriptors or no candidate produces a valid affine estimate.

At index time, overlapping grayscale tiles are generated from the 512-pixel working image with 64, 96, 128, 192, and 256-pixel square windows and a 50% stride. Each tile stores a 64-bit DCT perceptual hash, image ID, tile coordinates, and tile scale. At query time, vectorized Hamming distance retrieves 10 image candidates. Multi-scale grayscale and edge template matching then verifies only those candidates.

Fallback results use local appearance and candidate margin for ranking because geometric evidence is unavailable. Their similarity is capped at 89.9 so the score communicates weaker evidence without suppressing the required best result.

## API Contract

### `GET /api/status`

Returns:

```json
{
  "state": "ready",
  "indexed_images": 796,
  "build_time_ms": 12450,
  "error": null
}
```

`state` is one of `building`, `ready`, or `error`.

### `POST /api/match`

Accepts one multipart field named `file`.

Returns:

```json
{
  "query": {"width": 90, "height": 90},
  "elapsed_ms": 184,
  "matches": [
    {
      "image_id": "opaque-id",
      "parent_name": "dl_axiumcrisis",
      "filename": "1080_base.jpg",
      "width": 768,
      "height": 768,
      "similarity": 94.2,
      "image_url": "/api/images/opaque-id"
    }
  ]
}
```

The initial implementation returns exactly one item in `matches`. The array is retained as the extension point for Top-K results.

### `GET /api/images/{image_id}`

Returns the original indexed file with an appropriate media type and inline disposition. Unknown IDs return `404`.

## Upload Validation and Errors

- Read at most 10 MiB plus one byte; an overflow returns `413`.
- Treat MIME type and filename extension as hints only. OpenCV decode success determines whether input is an image.
- Reject empty or undecodable uploads with `400`.
- Reject decoded images above 25 megapixels with `413` to limit decompression-based memory use.
- Return `503` while the index is building or if startup failed.
- Return structured JSON errors with a short stable code and human-readable message.
- Log internal exceptions server-side and return a generic `500` response without exposing filesystem paths.

## WebUI

The visual language is a restrained dark gallery: charcoal background, warm white text, muted green status, and a small coral-red accent. The design avoids decorative hero copy and prioritizes the task.

### Upload State

- A compact header identifies the local matcher.
- Index state and indexed-image count appear directly above the upload control.
- The main content is one centered drag-and-drop area with a file picker.
- Supported formats and the 10 MiB limit are visible but secondary.
- While the index is building, upload is disabled and the status is polled.

### Result State

- The uploaded query appears as a small summary strip.
- Results are rendered as compact list rows, even though the first version receives one row.
- Each row contains the source thumbnail, parent directory, filename, dimensions, rank, similarity, and a link to the original image.
- There are no qualitative explanations such as geometric-inlier or edge-consistency text.
- A re-upload action returns to the upload state.

The production UI has no manual upload/result navigation tabs; those existed only in the review prototype. The layout collapses cleanly for mobile, keeps touch targets at least 44 pixels, supports keyboard upload, and honors `prefers-reduced-motion`.

## Test Strategy

### Unit Tests

- Source file inclusion and `*_256` exclusion.
- Deterministic catalog order and opaque ID resolution.
- Cache manifest equality and invalidation.
- Image decode, upload-byte limit, and decoded-pixel limit.
- Score normalization, clamping, and fallback cap.
- Safe rejection of unknown image IDs.

### Matching Tests

A deterministic benchmark command samples real source images with a fixed random seed. It generates color and grayscale square crops covering 10%-40% of the source's shorter edge and resizes them to 64, 90, 128, and 192 pixels. It records Top-1 correctness, whether primary or fallback matching was used, and candidate scores.

The initial acceptance target is at least 95% Top-1 accuracy over the generated fixture. Any failures are saved by source ID and crop coordinates so tuning is reproducible rather than based on isolated manual examples.

### API and Browser Tests

- Status during build, ready state, and failed startup.
- Successful multipart match and response schema.
- Empty, corrupt, oversized, and excessive-pixel uploads.
- Original-image retrieval and unknown-ID rejection.
- Upload, loading, result rendering, re-upload, and error recovery in the browser.
- Desktop and mobile responsive layouts.

### Performance Tests

The benchmark reports cold index-build duration, warm startup duration, cache size, process memory, and warm-query P50/P95 latency. The initial latency target is a P50 below one second on the development machine. Accuracy takes priority over reducing startup time, while persistent caching keeps repeated startup practical.

## Delivery Shape

The repository will contain a focused Python package for catalog, index, matcher, API schemas, and FastAPI setup; a static directory for the WebUI; tests and benchmark tooling; dependency metadata; and concise run instructions. Generated index files, uploads, Python caches, and benchmark failure artifacts are excluded from version control.
