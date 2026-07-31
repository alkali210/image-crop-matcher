# Runtime Gallery Selection and Top-3 Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a local WebUI user switch to any absolute server-local gallery without interrupting current searches, remember the successful directory, reuse per-gallery caches, and return the three strongest distinct matches.

**Architecture:** A runtime selection store and deterministic gallery cache namespace feed a thread-safe `GalleryManager` that owns immutable active bundles and atomically swaps them after background builds. `ImageMatcher.match_many` produces ordered distinct candidates while the existing `match` interface remains compatible. FastAPI exposes status/gallery endpoints and the native WebUI renders directory controls plus the existing result list.

**Tech Stack:** Python 3.11+, OpenCV 4.x, NumPy 2.x, FastAPI, Pydantic, native HTML/CSS/JavaScript, pytest.

## Global Constraints

- Directory input is an absolute path on the FastAPI host; do not upload or copy the directory through the browser.
- Persist only a successfully indexed directory in `.crop-matcher.json`; a missing file falls back to `songs/`.
- Keep the old bundle searchable until a new bundle is built and persisted, then swap catalog/index/matcher/path atomically.
- Permit only one pending gallery build; return `409` for another request while it runs.
- Store each gallery cache below `.cache/galleries/<sha256(normalized resolved absolute path)>/`.
- Return `min(3, gallery image count)` distinct results sorted by similarity descending and image ID for ties.
- Preserve `ImageMatcher.match()` and all existing API match-item fields.
- Preserve the current compact dark UI; no desktop directory dialog, `webkitdirectory`, Node build chain, CDN, or remote assets.
- Keep all current upload, path-containment, cache-integrity, and structured-error protections.

---

## File Structure

```text
.gitignore                              Ignore local persisted gallery selection
src/crop_matcher/config.py              Add selection-file setting
src/crop_matcher/gallery_state.py       Atomic selection store and per-gallery cache namespace
src/crop_matcher/gallery_manager.py     Active bundle, snapshots, build reservation, swap and rollback
src/crop_matcher/matcher.py             List-producing Top-K matching and compatibility wrapper
src/crop_matcher/schemas.py             Extended status and gallery request models
src/crop_matcher/main.py                Lifecycle, background switch endpoint, Top-3 serialization
src/crop_matcher/static/index.html      Absolute-path gallery controls
src/crop_matcher/static/styles.css      Directory block and responsive states
src/crop_matcher/static/app.js          Gallery submission/status polling and Top-3 rendering
tests/test_gallery_state.py              Persistence and cache namespace tests
tests/test_gallery_manager.py            Atomic switch, rollback and concurrency tests
tests/test_matcher.py                    Top-K ordering, fill, dedupe and compatibility tests
tests/test_api.py                        Gallery API, old-search availability and Top-3 response tests
README.md                                Directory switching, persistence, caches and Top-3 behavior
```

### Task 1: Persisted Selection and Gallery Cache Namespaces

**Files:**
- Modify: `.gitignore`
- Modify: `src/crop_matcher/config.py`
- Create: `src/crop_matcher/gallery_state.py`
- Create: `tests/test_gallery_state.py`

**Interfaces:**
- Produces: `GallerySelectionStore(path: Path)`, `load() -> Path | None`, `save(gallery_dir: Path) -> None`, and `gallery_cache_dir(cache_root: Path, gallery_dir: Path) -> Path`.
- Consumes: only filesystem and standard-library JSON/hash functions.

- [ ] **Step 1: Write failing persistence and namespace tests**

```python
from pathlib import Path

import pytest

from crop_matcher.gallery_state import GallerySelectionStore, gallery_cache_dir


def test_selection_store_round_trips_resolved_absolute_path(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    store = GallerySelectionStore(tmp_path / ".crop-matcher.json")
    assert store.load() is None
    store.save(gallery)
    assert store.load() == gallery.resolve()


def test_gallery_cache_namespaces_are_stable_and_distinct(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    assert gallery_cache_dir(tmp_path / ".cache", first) == gallery_cache_dir(
        tmp_path / ".cache", first
    )
    assert gallery_cache_dir(tmp_path / ".cache", first) != gallery_cache_dir(
        tmp_path / ".cache", second
    )


def test_malformed_selection_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / ".crop-matcher.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="selection"):
        GallerySelectionStore(path).load()
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `pytest tests/test_gallery_state.py -v`

Expected: collection fails because `crop_matcher.gallery_state` does not exist.

- [ ] **Step 3: Implement atomic selection and namespacing**

```python
# src/crop_matcher/gallery_state.py
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile


class GallerySelectionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Path | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text("utf-8"))
            value = payload["gallery_dir"]
            if not isinstance(value, str) or not value:
                raise ValueError
            gallery = Path(value)
            if not gallery.is_absolute():
                raise ValueError
            return gallery.resolve(strict=False)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("Invalid gallery selection file") from exc

    def save(self, gallery_dir: Path) -> None:
        resolved = gallery_dir.resolve(strict=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                json.dump({"gallery_dir": str(resolved)}, temporary, ensure_ascii=False)
                temporary_path = Path(temporary.name)
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def gallery_cache_dir(cache_root: Path, gallery_dir: Path) -> Path:
    normalized = os.path.normcase(str(gallery_dir.resolve(strict=True)))
    digest = sha256(normalized.encode("utf-8")).hexdigest()
    return cache_root / "galleries" / digest
```

Add `selection_file: Path = Path(".crop-matcher.json")` to `Settings` and append `.crop-matcher.json` to `.gitignore`.

- [ ] **Step 4: Run tests and static checks**

Run: `pytest tests/test_gallery_state.py -v && ruff check src tests && ruff format --check src tests`

Expected: all tests and checks pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore src/crop_matcher/config.py src/crop_matcher/gallery_state.py tests/test_gallery_state.py
git commit -m "feat: persist runtime gallery selection"
```

### Task 2: Thread-Safe Gallery Manager and Gallery API

**Files:**
- Create: `src/crop_matcher/gallery_manager.py`
- Modify: `src/crop_matcher/schemas.py`
- Modify: `src/crop_matcher/main.py`
- Create: `tests/test_gallery_manager.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `GallerySelectionStore`, `gallery_cache_dir`, `ImageCatalog`, `FeatureIndex`, and `ImageMatcher`.
- Produces: immutable `GalleryBundle`, `GallerySnapshot`, `GalleryManager.initialize(gallery_dir: Path | None = None)`, `reserve_switch(path)`, `run_reserved_switch(path)`, and extended status/gallery API.

- [ ] **Step 1: Write failing manager tests**

Construct managers with `GalleryManager(settings, selection_store, builder=fake_builder)`. The fake builder returns `GalleryBundle` objects with small real catalogs/indexes; the blocking variant waits on two `threading.Event` objects before returning the second bundle.

```python
def test_failed_replacement_keeps_active_bundle(manager, first_gallery, broken_gallery) -> None:
    manager.initialize(first_gallery)
    old = manager.snapshot().active
    manager.reserve_switch(broken_gallery)
    manager.run_reserved_switch(broken_gallery)
    snapshot = manager.snapshot()
    assert snapshot.active is old
    assert snapshot.pending_gallery_dir is None
    assert snapshot.switch_error


def test_replacement_keeps_old_bundle_until_atomic_swap(
    manager, first_gallery, second_gallery, blocking_builder
) -> None:
    manager.initialize(first_gallery)
    old = manager.snapshot().active
    manager.reserve_switch(second_gallery)
    worker = blocking_builder.start(manager.run_reserved_switch, second_gallery)
    assert manager.snapshot().active is old
    assert manager.snapshot().reindexing is True
    blocking_builder.release()
    worker.join()
    assert manager.snapshot().active.gallery_dir == second_gallery.resolve()
```

Add API tests for invalid relative/missing paths (`400`), a pending build (`409`), same active path (`200`), and accepted switch (`202`).

- [ ] **Step 2: Run focused tests to verify RED**

Run: `pytest tests/test_gallery_manager.py tests/test_api.py -k "gallery or replacement" -v`

Expected: imports/routes fail because manager and gallery API do not exist.

- [ ] **Step 3: Implement immutable bundles and manager state transitions**

Use these exact public domain types:

```python
@dataclass(frozen=True, slots=True)
class GalleryBundle:
    gallery_dir: Path
    cache_dir: Path
    catalog: ImageCatalog
    feature_index: FeatureIndex
    matcher: ImageMatcher
    build_time_ms: int


@dataclass(frozen=True, slots=True)
class GallerySnapshot:
    state: Literal["building", "ready", "error"]
    active: GalleryBundle | None
    pending_gallery_dir: Path | None
    switch_error: str | None
    initial_error: str | None

    @property
    def reindexing(self) -> bool:
        return self.active is not None and self.pending_gallery_dir is not None
```

`GalleryManager(settings, selection_store, builder=None)` accepts an injectable builder with signature `(gallery_dir: Path, cache_dir: Path) -> GalleryBundle`; production uses the real catalog/index/matcher builder. `initialize(gallery_dir=None)` loads the persisted selection when no explicit path is supplied, then falls back to `settings.gallery_dir` only when no selection exists. `reserve_switch` validates and resolves the path, returns `"active"` for the current path, raises a manager conflict when pending, otherwise stores the pending path under a lock. `run_reserved_switch` builds outside the lock, saves selection before activation, then swaps the complete bundle under the lock. Any exception records a safe switch error and retains the old bundle.

Extend schemas:

```python
class GalleryRequest(BaseModel):
    path: str


class StatusResponse(BaseModel):
    state: Literal["building", "ready", "error"]
    indexed_images: int
    build_time_ms: int | None
    error: str | None
    gallery_dir: str | None
    pending_gallery_dir: str | None
    reindexing: bool
    switch_error: str | None
```

In `main.py`, initial lifespan runs `manager.initialize()` in `asyncio.to_thread`. `POST /api/gallery` calls `reserve_switch` synchronously and, when accepted, creates one tracked `asyncio.create_task(asyncio.to_thread(manager.run_reserved_switch, path))`. Await tracked build tasks during shutdown. Every existing endpoint takes one manager snapshot and uses `snapshot.active` only.

- [ ] **Step 4: Run manager/API regression tests**

Run: `pytest tests/test_gallery_manager.py tests/test_api.py -v && ruff check src tests`

Expected: path validation, background switch, rollback, old-bundle availability, status fields, and previous API tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/crop_matcher/gallery_manager.py src/crop_matcher/schemas.py src/crop_matcher/main.py tests/test_gallery_manager.py tests/test_api.py
git commit -m "feat: switch galleries without search downtime"
```

### Task 3: Top-3 Matcher and API Results

**Files:**
- Modify: `src/crop_matcher/matcher.py`
- Modify: `src/crop_matcher/main.py`
- Modify: `tests/test_matcher.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: current SIFT `CandidateScore`, fallback tile scores, `GalleryBundle.matcher`.
- Produces: `ImageMatcher.match_many(query_bgr: np.ndarray, limit: int = 3) -> list[MatchResult]`; `match()` remains `MatchResult`.

- [ ] **Step 1: Add failing Top-K tests**

```python
def test_match_many_returns_three_distinct_results_sorted_by_similarity(matcher, query) -> None:
    results = matcher.match_many(query, limit=3)
    assert len(results) == 3
    assert len({result.record.image_id for result in results}) == 3
    assert [result.similarity for result in results] == sorted(
        (result.similarity for result in results), reverse=True
    )


def test_match_many_returns_all_results_when_gallery_has_fewer_than_limit(tiny_matcher, query) -> None:
    results = tiny_matcher.match_many(query, limit=3)
    assert len(results) == 2


def test_match_remains_first_match_many_result(matcher, query) -> None:
    assert matcher.match(query) == matcher.match_many(query, limit=1)[0]
```

Add an API test with three deterministic source records asserting `len(body["matches"]) == 3`, unique IDs, descending similarities, and rank-preserving array order.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_matcher.py tests/test_api.py -k "match_many or three" -v`

Expected: `ImageMatcher` has no `match_many`, and API returns one result.

- [ ] **Step 3: Refactor candidate production without changing first-result behavior**

Extract primary candidate scoring into `_primary_results(...) -> list[MatchResult]` and fallback scoring into `_fallback_results(query_gray, exclude_ids) -> list[MatchResult]`. Compute one method-level margin bonus and apply it equally to all results from that method, sort by `(-similarity, image_id)`, merge by image ID, and return `min(limit, len(catalog.records))`.

Use these wrappers:

```python
def match(self, query_bgr: np.ndarray) -> MatchResult:
    return self.match_many(query_bgr, limit=1)[0]


def match_many(self, query_bgr: np.ndarray, limit: int = 3) -> list[MatchResult]:
    if limit < 1:
        raise ValueError("limit must be positive")
    # Extract query features once, gather SIFT then fallback candidates,
    # deduplicate, sort, and slice to min(limit, gallery size).
```

Keep SIFT and FLANN locks, geometry validation, pHash cap, and deterministic candidate fill unchanged. `main.py` calls `match_many(query, limit=3)` once in the worker thread and serializes each result.

- [ ] **Step 4: Run matcher/API and benchmark regressions**

Run: `pytest tests/test_matcher.py tests/test_api.py -v`

Expected: Top-3 tests and all existing first-result benchmark helpers pass.

- [ ] **Step 5: Commit**

```bash
git add src/crop_matcher/matcher.py src/crop_matcher/main.py tests/test_matcher.py tests/test_api.py
git commit -m "feat: return top three image matches"
```

### Task 4: Gallery Controls, Top-3 UI, Documentation and Integration

**Files:**
- Modify: `src/crop_matcher/static/index.html`
- Modify: `src/crop_matcher/static/styles.css`
- Modify: `src/crop_matcher/static/app.js`
- Modify: `tests/test_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: extended `/api/status`, `POST /api/gallery`, and Top-3 `matches` array.
- Produces: accessible absolute-path gallery control, nonblocking reindex status, failure recovery, and three compact result rows.

- [ ] **Step 1: Add failing static/API contract tests**

```python
def test_webui_contains_runtime_gallery_controls(client) -> None:
    html = client.get("/").text
    assert 'id="gallery-path"' in html
    assert 'id="switch-gallery-button"' in html
    assert 'id="gallery-switch-status"' in html


def test_frontend_submits_gallery_path_and_keeps_upload_during_reindex(client) -> None:
    script = client.get("/static/app.js").text
    assert 'fetch("/api/gallery"' in script
    assert "pending_gallery_dir" in script
    assert "reindexing" in script
```

- [ ] **Step 2: Run the shell tests to verify RED**

Run: `pytest tests/test_api.py -k "gallery_controls or reindex" -v`

Expected: required controls and JavaScript behavior are absent.

- [ ] **Step 3: Implement the approved UI behavior**

Add a labeled text input with `autocomplete="off"`, current-path output, switch button, and `aria-live` switch status below the index status. Extend `state` with `galleryPending`. Submit trimmed path as JSON; disable only the switch control while pending. Status polling continues during `reindexing`, updates active/pending paths, and keeps upload enabled whenever an active gallery is ready. On switch failure, show the directory-local error while preserving upload behavior.

Keep `renderMatches` unchanged except for accepting the three-item array already produced by the API. The existing `createMatchRow` rank argument renders `匹配 #1` through `匹配 #3`.

Add responsive CSS so the path input and button share one row on desktop and stack below `700px`; use existing colors, borders, spacing, focus styles, and 44px controls.

Document absolute-path selection, `.crop-matcher.json`, per-gallery cache namespaces, background swap/rollback, `409`, and Top-3 output in `README.md`.

- [ ] **Step 4: Run complete verification and browser checks**

Run: `pytest -v && ruff check . && ruff format --check . && node --check src/crop_matcher/static/app.js && git diff --check`

Expected: all tests and checks pass.

Browser checks:

- Desktop: current path, input and button are compact and aligned below index status.
- Mobile `390x844`: controls stack without overflow and retain 44px targets.
- Switch to a second gallery: old search remains usable while pending, then status and results use the new gallery.
- Submit an invalid/no-image directory: local error appears and old gallery still returns results.
- Search a gallery with at least three images: three ranked compact rows appear with valid original links.

- [ ] **Step 5: Commit**

```bash
git add src/crop_matcher/static README.md tests/test_api.py
git commit -m "feat: add runtime gallery controls"
```

## Final Review Checklist

- [ ] `.crop-matcher.json` is ignored and only written after a successful build.
- [ ] Gallery path validation and cache namespace hashing use resolved absolute paths.
- [ ] Replacement build failure leaves the active bundle and search behavior unchanged.
- [ ] In-flight search/image requests retain a valid old bundle snapshot during swap.
- [ ] Top-3 results are distinct, deterministic, sorted, and limited by gallery size.
- [ ] Existing `match()` and benchmark Top-1 behavior remain covered.
- [ ] Full tests, static checks, desktop/mobile browser flows, and Git cleanliness pass.
