# Image Crop Matcher

Image Crop Matcher is a local crop-retrieval tool. It finds the source image for a crop within a selected gallery using traditional computer-vision algorithms, without neural models or remote recognition services.

## Features

- Recursively indexes `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` images in a local gallery.
- Returns up to 5 distinct source-image candidates for each uploaded crop, ranked by similarity and linked to the original files.
- Switches gallery paths from the WebUI; a new gallery is indexed in the background while the current gallery remains available.
- Supports repeated uploads and light/dark themes that follow the browser preference.
- Uses `gallery/` as the default gallery. Images whose filename stem ends in `_256` are treated as generated thumbnails and excluded.
- Limits each upload to 10 MiB and each decoded image to 25 MP.

The matcher is intended for regions cropped from gallery images and then resized. Rotation, mirroring, perspective transforms, borders, watermarks, and interface overlays are not supported transformations.

## Usage

Run the following commands from the repository root:

```bash
python -m pip install uv
uv sync
```

Place source images under `gallery/`, then start the service:

```bash
uv run uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in a browser. The left column accepts another existing gallery as an absolute path. A successful selection is stored in `.crop-matcher.json` and reused on the next startup.

Index caches are stored under `.cache/galleries/` as independent `.npy` arrays. Full SIFT descriptors use lossless `uint8` storage and are memory-mapped with their required keypoint coordinates, so only verified candidates page their local features into memory. A compact global FLANN index keeps up to 128 spatially distributed float descriptors per image instead of indexing every descriptor. The low-resolution template pixels are memory-mapped as well. The application does not continuously watch gallery files. After adding, replacing, moving, or deleting images, restart the service or switch to another gallery and back in the WebUI.

## Matching Method

During indexing, the application extracts lossless `uint8` SIFT features, selects a spatially balanced representative set for global retrieval, and builds a low-resolution grayscale template pyramid. Queries use the following two matching paths.

### SIFT and Geometric Verification

1. Convert the query to grayscale. If its shortest edge is below 256 px, scale it proportionally to 256 px before extracting SIFT keypoints and descriptors.
2. Use FLANN KNN with a Lowe ratio of `0.78` to vote for source-image candidates from the compact representative descriptor index. Rematch each candidate against its memory-mapped full `uint8` descriptors with exact BF L2 distance. If this does not produce enough geometrically valid candidates, supplement it with the low-resolution template shortlist.
3. Use the local matches to run deterministic RANSAC constrained to uniform scale and translation. At least 4 geometric inliers are required; rotation is not part of the fitted model.
4. Refine the SIFT scale and translation with a local pixel-correlation search, then warp the candidate region to the query dimensions.
5. Calculate grayscale and gradient correlations. SIFT inliers validate and locate the crop, then provide a bounded geometric correction to the pixel score.

Normalized correlations are clipped to `[0, 1]`, so zero or negative correlation contributes no evidence. The SIFT path uses:

```text
edge_weight          = 0.1 + 0.2 × structure_reliability
appearance           = (1 - edge_weight) × gray_score + edge_weight × edge_score
count_quality        = min(inlier_count / 12, 1)
reprojection_quality = clip(1 - mean_reprojection_error / 4, 0, 1)
spread_quality       = normalized x/y span of query inliers
geometry_quality     = 0.5 × inlier_ratio
                     + 0.1 × count_quality
                     + 0.25 × reprojection_quality
                     + 0.15 × spread_quality
geometry_confidence  = clip((geometry_quality - 0.5) / 0.4, 0, 1)
structure_reliability = clip(query_edge_density / 0.35, 0, 1)
geometry_weight       = 0.35 - 0.25 × structure_reliability
similarity            = 100 × clip(
    appearance × (1 + 2 × geometry_weight × geometry_confidence
                  × (1 - appearance)),
    0,
    1
)
```

The query edge density is calculated once from the gradient image using a threshold of `64`. Sparse-structure queries use up to `35%` geometry weight and reduce edge correlation to `10%`, while texture-rich queries limit geometry to `10%` and use `30%` edge correlation. Geometry is confirmation-only: a valid model can increase but never decrease the pixel score. The `(1 - appearance)` term preserves headroom for strong pixel matches instead of saturating them at `100%`. Degenerate inliers spanning less than `2%` of either query axis cannot produce a positive geometric correction. Because geometry multiplies appearance, zero pixel evidence always remains zero.

### Template Fallback

The template path supplements results when SIFT evidence is insufficient, the query's shortest edge is below 64 px, or SIFT does not produce enough entries to fill the candidate list:

1. Coarsely rank candidates with normalized cross-correlation (NCC) over the low-resolution grayscale template pyramid. Reuse this shortlist if SIFT supplementation already computed it.
2. Match grayscale and gradient images at multiple scales for only as many leading candidates as the response still needs.
3. Calculate the ranking score as follows:

```text
edge_weight    = 0.1 + 0.2 × structure_reliability
template_score = (1 - edge_weight) × gray_score + edge_weight × edge_score
similarity     = 100 × template_score
```

Template-only candidates have no geometric evidence and keep their unmodified pixel score. For sparse queries, a geometrically verified candidate also checks its template alignment and retains whichever complete hypothesis produces the stronger score; geometry from one location is never applied to a different template location.

Final results are deduplicated and sorted by descending `similarity`. The value is a ranking-confidence score, not a probability. There is no rejection threshold, so the matcher returns available gallery candidates even when evidence is weak.

## API

The base URL is `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `/docs`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/status` | Returns index state, image count, build time, and gallery-switch state. |
| `POST` | `/api/gallery` | Switches to a gallery identified by an absolute path. |
| `POST` | `/api/match` | Uploads a query image and returns up to 5 candidates. |
| `GET` | `/api/images/{gallery_id}/{image_id}` | Returns an original image referenced by a match. |

### Status and Gallery

`GET /api/status` returns `state`, `indexed_images`, `build_time_ms`, active and pending gallery paths, `reindexing`, and error fields. `state` is `building`, `ready`, or `error`.

`POST /api/gallery` accepts JSON:

```json
{"path":"C:\\images\\gallery"}
```

A newly accepted background build returns `202`; an already active target returns `200`. An invalid path returns `400`. Initial indexing or another gallery switch in progress returns `409`.

### Image Matching

`POST /api/match` requires a multipart field named `file`:

```powershell
curl.exe -F "file=@C:\path\to\crop.png" http://127.0.0.1:8000/api/match
```

```json
{
  "query": {"width": 128, "height": 128},
  "elapsed_ms": 42,
  "matches": [{
    "image_id": "9656ad30b81ad6edd501eec4",
    "parent_name": "song-name",
    "filename": "cover.jpg",
    "width": 768,
    "height": 768,
    "similarity": 97.4,
    "image_url": "/api/images/gallery-id/9656ad30b81ad6edd501eec4"
  }]
}
```

Use `image_url` directly to display or download the corresponding source image.

### Error Format

API errors use the following envelope:

```json
{"error":{"code":"invalid_image","message":"..."}}
```

Common status codes are `400` for invalid images or paths, `409` for gallery-switch conflicts, `413` for upload or decoded-image size limits, and `503` while the index is unavailable.
