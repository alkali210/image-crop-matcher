# Image Crop Matcher

Image Crop Matcher is a local FastAPI application that finds the source image for a resized
square crop. SIFT with geometric verification is the primary matching path. When SIFT evidence is
insufficient, or when a query is smaller than the practical 64-pixel feature boundary,
low-resolution coarse grayscale normalized cross-correlation (NCC) retrieves candidates for full
grayscale and gradient template refinement. The application does not use neural models or remote
recognition services.

## Windows Setup

Install uv, then run these commands from the repository root in Command Prompt or PowerShell:

```bat
uv sync --extra dev
uv run uvicorn crop_matcher.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. Keep the terminal open while using the application. Run tests,
lint checks, formatting checks, and the example regression through uv with:

```bat
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run benchmarks/example_regression.py --gallery songs --examples examples
```

The same uv commands work on macOS and Linux; no environment activation is required.

## Gallery And Startup

Place source images anywhere below `songs/`, or enter another existing **absolute directory path**
in the web interface. The recursive scan supports `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`
files, case-insensitively. Files whose stem ends in `_256` are treated as generated thumbnails and
excluded.

The first use of a gallery scans and decodes it, then stores catalog metadata, SIFT descriptors,
and coarse grayscale template arrays in a per-gallery namespace below `.cache/galleries/`. The
namespace is derived from the gallery's resolved absolute path, so two galleries never overwrite
each other's cache. Later uses compare file paths, sizes, and modification times before restoring
cached dimensions and feature arrays; source image contents are not read again while that gallery
is unchanged. The in-memory FLANN search tree is still rebuilt from the cached descriptors.
`GET /api/status` reports `building`, `ready`, or `error`, the indexed image count and build time,
and active/pending gallery paths.

Cache schema changes are detected automatically. After upgrading from an older schema, the next
startup performs one full rebuild; subsequent starts use the new catalog and feature caches
normally.

After a successful runtime switch, the selected absolute path is saved in `.crop-matcher.json` and
used on the next startup. This local state file is ignored by Git and is not written when indexing
the replacement fails. A switch builds in the background: the previous gallery remains available
for uploads, searches, and original-image links until the replacement is complete, then the active
bundle is swapped atomically. A failed build rolls back to the previous gallery and displays an
error beside the path control.

The application does not watch an active gallery for file changes. To rebuild after adding,
replacing, moving, or removing images, restart Uvicorn or switch away and back. A manifest change
invalidates that gallery's feature cache.

## Queries And Results

Supported queries are square regions cropped from a gallery source, resized to a different square
size, and optionally converted to grayscale. Rotation, mirroring, perspective changes, borders,
watermarks, and interface overlays are not supported transforms. The practical minimum expected
side length is 64 pixels.

Uploads may use any of the five supported image formats regardless of filename or declared MIME
type; successful OpenCV decoding determines validity. Each upload is limited to 10 MiB and its
decoded dimensions are limited to 25 megapixels.

The service returns up to three distinct sources in descending rank because a valid query is
expected to come from the gallery. Smaller galleries return fewer results. It does not use a
rejection threshold. `similarity` is a bounded 0-100 ranking-confidence score, not a probability.
Template-refinement scores are capped at 89.9 to indicate weaker geometric evidence. Identical
source regions and nearly featureless crops are inherently ambiguous, so the highest-ranked result
may not be the intended source in those cases.

## API

The application exposes four API routes:

- `GET /api/status` returns startup state, indexed image count, build time, active and pending paths,
  reindexing state, and any public startup or switch error.
- `POST /api/gallery` accepts JSON such as `{"path":"C:\\images\\gallery"}`. The path must be
  absolute. It returns `202` while a replacement builds, `200` when that gallery is already active,
  `400` for an invalid path, or `409` when another startup/build switch is in progress.
- `POST /api/match` accepts one multipart field named `file` and returns query dimensions, elapsed
  time, and up to three ranked entries in `matches`.
- `GET /api/images/{gallery_id}/{image_id}` returns the trusted original gallery file for a
  namespaced URL returned by the matcher. Links remain available for the active and immediately
  previous gallery; after later successful switches, older links safely expire with `404` instead
  of resolving against another gallery.

Example PowerShell request:

```powershell
curl.exe -F "file=@C:\path\to\crop.png" http://127.0.0.1:8000/api/match
```

While the initial startup is still building, match and image requests return `503`. During a runtime
replacement build they continue using the active gallery. Invalid images return `400`; oversized
files or decoded images return `413`. Concurrent gallery switch requests return `409`. API errors use
`{"error":{"code":"...","message":"..."}}`.

## Benchmark

The benchmark generates a deterministic crop workload with seed `20260730`: color and grayscale
crops cover 10%-40% of each source's shorter edge and use output sizes of 64, 90, 128, or 192 pixels.
It also seeds OpenCV's RNG when that API is available. OpenCV FLANN and RANSAC are approximate,
however, so repeated runs can have small prediction variations even though the crop specifications
are identical. Do not expect byte-identical failure sets across OpenCV builds or machines.

The command builds one catalog, index, and matcher, performs one untimed warm-up, and reports Top-1
accuracy, SIFT/template usage, and matcher-only P50/P95 latency.

Start with a bounded smoke run before the complete 796-image run:

```bat
uv run benchmarks/benchmark.py --gallery songs --samples-per-image 1 --max-images 8
uv run benchmarks/benchmark.py --gallery songs --samples-per-image 4 --seed 20260730
```

### Previous pHash Baseline

The following 73.02% result is retained as the previous pHash baseline. It is not a measurement of
the current template-retrieval implementation and was an **accepted initial limitation**, not a
passing 95% result:

```text
images=796 queries=3184
top1=2325/3184 accuracy=73.02%
method_sift=2586 method_phash=598
latency_ms_p50=49.436 latency_ms_p95=305.458
```

That previous run recorded 2,325/3,184 correct matches and 859 failures, producing one
incorrect-result file per failure.

### Current Template-Retrieval Baseline

The full uv benchmark on the same 796-image gallery improved Top-1 accuracy by 3.30 percentage
points while keeping matcher P95 below one second:

```text
images=796 queries=3184
top1=2430/3184 accuracy=76.32%
method_sift=2588 method_template=596
latency_ms_p50=49.300 latency_ms_p95=708.004
```

The five supplied 57x57 grayscale regression examples all matched their expected source; their
measured P95 was 712.299 ms. The coarse template arrays occupy 32.24 MiB for this gallery. The full
benchmark still exits with status `1` because 76.32% remains below the aspirational 95% threshold.

Available options are `--gallery`, `--samples-per-image`, `--max-images`, `--seed`, and
`--failure-dir`. A bounded `--max-images` run stores and reuses its index under a deterministic
`.cache/benchmarks/bounded-<digest>/` namespace based on its bounded catalog and feature settings. It
never reads or replaces the full-gallery manifest and arrays directly under `.cache/`.

Incorrect matches are written below a deterministic `run-<digest>/` subdirectory of
`benchmark-failures/` by default. Each run directory contains a benchmark ownership marker. At the
start of a run, before gallery scanning or warm-up, the benchmark deletes only its own
`failure-*.json` files directly inside that marked run directory. It does not delete matching files
from the caller-provided root or other run directories, and it refuses to reuse an unmarked
directory. Each failure file records source and predicted IDs, crop coordinates and side, output
size, grayscale state, similarity, and measured latency.

The command exits with status `1` only when Top-1 accuracy is below 95%. Latency is reported for
operator comparison but never changes the process status because it depends on the machine and
current load. Build, decode, match, and file-write errors are not hidden; they terminate the command
with their normal nonzero error status.
