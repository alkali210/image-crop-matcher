# Image Crop Matcher

Image Crop Matcher is a local FastAPI application that finds the source image for a resized
square crop. It uses SIFT, geometric verification, local appearance scoring, and a perceptual-hash
fallback. It does not use neural models or remote recognition services.

## Windows Setup

Run these commands from the repository root in Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000
```

For PowerShell, activate the same environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then open <http://127.0.0.1:8000>. Keep the terminal open while using the application. Run tests
and the accepted full benchmark from an activated environment with:

```bat
pytest -v
python benchmarks/benchmark.py --gallery songs --samples-per-image 4
```

On macOS or Linux, the Python commands are unchanged; activate with
`source .venv/bin/activate` instead.

## Gallery And Startup

Place source images anywhere below `songs/`. The recursive scan supports `.jpg`, `.jpeg`, `.png`,
`.webp`, and `.bmp` files, case-insensitively. Files whose stem ends in `_256` are treated as
generated thumbnails and excluded.

The first startup scans the gallery and builds SIFT and perceptual-hash data in `.cache/`, so it
takes longer. Later startups validate the gallery manifest and load cached NumPy arrays with
pickling disabled; the in-memory FLANN search tree is still rebuilt. `GET /api/status` reports
`building`, `ready`, or `error` and includes the indexed image count and build time.

The application does not watch the gallery. Restart Uvicorn after adding, replacing, moving, or
removing an image. A gallery manifest change invalidates the complete feature cache, which is then
rebuilt during startup.

## Queries And Results

Supported queries are square regions cropped from a gallery source, resized to a different square
size, and optionally converted to grayscale. Rotation, mirroring, perspective changes, borders,
watermarks, and interface overlays are not supported transforms. The practical minimum expected
side length is 64 pixels.

Uploads may use any of the five supported image formats regardless of filename or declared MIME
type; successful OpenCV decoding determines validity. Each upload is limited to 10 MiB and its
decoded dimensions are limited to 25 megapixels.

The service returns the highest-ranked source because a valid query is expected to come from the
gallery. It does not use a rejection threshold. `similarity` is a bounded 0-100 ranking-confidence
score, not a probability. Perceptual-hash fallback scores are capped at 89.9 to indicate weaker
geometric evidence. Identical source regions and nearly featureless crops are inherently ambiguous,
so the guaranteed best result may not be the intended source in those cases.

## API

The application exposes three API routes:

- `GET /api/status` returns startup state, indexed image count, build time, and any public startup
  error.
- `POST /api/match` accepts one multipart field named `file` and returns query dimensions, elapsed
  time, and exactly one current best match in `matches`.
- `GET /api/images/{image_id}` returns the trusted original gallery file for an ID returned by the
  matcher. Unknown IDs return `404`.

Example PowerShell request:

```powershell
curl.exe -F "file=@C:\path\to\crop.png" http://127.0.0.1:8000/api/match
```

While startup is still building, match and image requests return `503`. Invalid images return
`400`; oversized files or decoded images return `413`. API errors use
`{"error":{"code":"...","message":"..."}}`.

## Benchmark

The benchmark generates deterministic color and grayscale crops with seed `20260730`, crop sides
covering 10%-40% of each source's shorter edge, and output sizes of 64, 90, 128, or 192 pixels. It
builds one catalog, index, and matcher, performs one untimed warm-up, and reports Top-1 accuracy,
SIFT/pHash usage, and matcher-only P50/P95 latency.

Start with a bounded smoke run before the complete 796-image run:

```bat
python benchmarks/benchmark.py --gallery songs --samples-per-image 1 --max-images 8
python benchmarks/benchmark.py --gallery songs --samples-per-image 4 --seed 20260730
```

Available options are `--gallery`, `--samples-per-image`, `--max-images`, `--seed`, and
`--failure-dir`. Incorrect matches are written as deterministic `failure-XXXXXX.json` files under
`benchmark-failures/` by default. Each file records source and predicted IDs, crop coordinates and
side, output size, grayscale state, similarity, and measured latency. Existing generated failure
files in that directory are cleared at the beginning of a run.

The command exits with status `1` only when Top-1 accuracy is below 95%. Latency is reported for
operator comparison but never changes the process status because it depends on the machine and
current load. Build, decode, match, and file-write errors are not hidden; they terminate the command
with their normal nonzero error status.
