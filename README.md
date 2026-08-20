# Image Crop Matcher

Image Crop Matcher is a local crop-retrieval tool. It finds the source image for a crop within a selected gallery using traditional computer-vision algorithms, without neural models or remote recognition services.

## Features

- Recursively indexes `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` images in a local gallery.
- Returns up to 5 distinct source-image candidates for each uploaded crop, ranked by similarity and linked to the original files.
- Switches gallery paths from the WebUI; a new gallery is indexed in the background while the current gallery remains available.
- Supports repeated uploads and light/dark themes that follow the browser preference.
- Uses `songs/` as the default gallery. Images whose filename stem ends in `_256` are treated as generated thumbnails and excluded.
- Limits each upload to 10 MiB and each decoded image to 25 MP.

The matcher is intended for regions cropped from gallery images and then resized. Rotation, mirroring, perspective transforms, borders, watermarks, and interface overlays are not supported transformations.

## Usage

Run the following commands from the repository root:

```bash
python -m pip install uv
uv sync
```

Place source images under `songs/`, then start the service:

```bash
uv run uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in a browser. The left column accepts another existing gallery as an absolute path. A successful selection is stored in `.crop-matcher.json` and reused on the next startup.

Index caches are stored under `.cache/galleries/`. The application does not continuously watch gallery files. After adding, replacing, moving, or deleting images, restart the service or switch to another gallery and back in the WebUI.

## Matching Method

During indexing, the application extracts SIFT features from each image and builds a FLANN matcher and a low-resolution grayscale template pyramid. Queries use the following two matching paths.

### SIFT and Geometric Verification

1. Convert the query to grayscale. If its shortest edge is below 256 px, scale it proportionally to 256 px before extracting SIFT keypoints and descriptors.
2. Use FLANN KNN with a Lowe ratio of `0.78` to vote for source-image candidates from the global descriptor index.
3. Rematch descriptors for each candidate and estimate a partial affine transform with RANSAC. At least 4 geometric inliers are required.
4. Warp the candidate region to the query dimensions and calculate grayscale and gradient correlations.

Normalized correlations are mapped to `[0, 1]`. The primary score is:

```text
appearance           = 0.7 × gray_score + 0.3 × edge_score
count_quality        = min(inlier_count / 20, 1)
reprojection_quality = clip(1 - mean_reprojection_error / 4, 0, 1)
geometry              = clip(0.5 × inlier_ratio + 0.3 × count_quality + 0.2 × reprojection_quality, 0, 1)
raw_score             = 0.4 × geometry + 0.5 × appearance
margin                = clip((best_score - second_score) / 0.2, 0, 1)
similarity            = 100 × clip(raw_score + 0.1 × margin, 0, 1)
```

### Template Fallback

The template path supplements results when SIFT evidence is insufficient, the query's shortest edge is below 64 px, or SIFT does not produce enough entries to fill the candidate list:

1. Coarsely rank candidates with normalized cross-correlation (NCC) over the low-resolution grayscale template pyramid.
2. Match grayscale and gradient images at multiple scales of each candidate source.
3. Calculate the ranking score as follows:

```text
template_score = 0.7 × gray_score + 0.3 × edge_score
similarity     = min(89.9, 100 × (0.85 × template_score + 0.15 × margin))
```

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
{"path":"C:\\images\\songs"}
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
