# Final Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all whole-branch review findings with bounded image decoding, contained catalog paths, complete fallback evidence, strict response contracts, and thread-safe matching.

**Architecture:** Pillow performs lazy header inspection before OpenCV receives bytes, while all decode-library failures are normalized to domain exceptions. `ImageCatalog.get` becomes the containment boundary used by index construction, matching, and file delivery; the feature cache guarantees at least one tile owner per indexed image. Pydantic and the browser independently validate successful response data, and existing matcher locks are extended to shared SIFT state.

**Tech Stack:** Python 3.11+, Pillow, OpenCV, NumPy, FastAPI, Pydantic v2, pytest, vanilla JavaScript, Node.js.

## Global Constraints

- Work only in `D:\workspace\auto-guessc\.worktrees\image-crop-matcher`.
- Use RED/GREEN TDD for every behavior change.
- Create one coherent fix commit and do not amend.
- Do not run the expensive full benchmark; run a bounded benchmark only if smoke validation is needed.
- Write the final report to `D:\workspace\auto-guessc\.git\worktrees\image-crop-matcher\sdd\final-fix-report.md`.

---

### Task 1: Header-Only Decode Preflight And Catalog Decode Isolation

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/crop_matcher/imaging.py`
- Test: `tests/test_imaging.py`
- Test: `tests/test_api.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: encoded bytes and `max_pixels: int`.
- Produces: `decode_image_bytes(data: bytes, max_pixels: int) -> np.ndarray`, raising only `ImageDecodeError` or `ImageTooLargeError` for decode/preflight failures.

- [ ] **Step 1: Write failing tests**

Add tests that monkeypatch `cv2.imdecode` and prove an oversized compressed PNG raises `ImageTooLargeError` without calling OpenCV, the API maps it to structured 413, OpenCV exceptions become `ImageDecodeError`, and catalog scanning skips one OpenCV-failing source while retaining valid peers.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_imaging.py tests/test_catalog.py tests/test_api.py -v`

Expected: failures show OpenCV called before the pixel limit and `cv2.error` escaping catalog scan.

- [ ] **Step 3: Implement minimal preflight**

Add Pillow as a runtime dependency. In `decode_image_bytes`, lazily call `Image.open(BytesIO(data))`, inspect declared width and height under a decompression-bomb warning filter, map bomb conditions and excessive dimensions to `ImageTooLargeError`, map invalid headers to `ImageDecodeError`, then call `cv2.imdecode` and normalize `cv2.error`/`None` to `ImageDecodeError`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_imaging.py tests/test_catalog.py tests/test_api.py -v`

Expected: all focused tests pass.

### Task 2: Catalog Path Containment

**Files:**
- Modify: `src/crop_matcher/catalog.py`
- Modify: `src/crop_matcher/feature_index.py`
- Modify: `src/crop_matcher/main.py`
- Test: `tests/test_catalog.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: lexical gallery candidates and image IDs.
- Produces: `ImageCatalog.get(image_id: str) -> ImageRecord` whose returned `path` is freshly resolved, regular, and under the resolved gallery root; unsafe records raise `KeyError`.

- [ ] **Step 1: Write failing tests**

Add deterministic tests with an outside-path record, privilege-conditional real symlink tests for scan and post-scan replacement, and an API test that expects 404 when lookup containment rejects the record.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_catalog.py tests/test_api.py -v`

Expected: outside records resolve and the image endpoint exposes their path.

- [ ] **Step 3: Implement containment boundary**

Resolve every scan candidate before decoding and skip targets outside `root.resolve()`. Preserve the lexical candidate in records so replacement is detectable, re-resolve it in `get`, verify containment and regular-file status, and return a copied record containing the safe resolved path. Make feature construction retrieve each source through `catalog.get`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_catalog.py tests/test_api.py tests/test_feature_index.py tests/test_matcher.py -v`

Expected: all containment and consumers pass.

### Task 3: Complete Tiny-Image Fallback Evidence

**Files:**
- Modify: `src/crop_matcher/feature_index.py`
- Test: `tests/test_feature_index.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: at least one fallback tile per indexed image, using a `(0, 0, min(width, height))` square when no configured tile fits.

- [ ] **Step 1: Write failing tests**

Change the small-image cache test to require one typed persisted tile, add a cache-validation regression for a missing image owner, and add an API lifecycle/match test for one tiny blank source.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_feature_index.py tests/test_api.py -v`

Expected: tiny tile arrays are empty and `/api/match` returns 500.

- [ ] **Step 3: Implement fallback tile and invariant**

Track whether each image emitted a configured tile; otherwise append one minimum-side square hash. Increment `CACHE_SCHEMA_VERSION` and reject archives whose tile owners do not cover every image index.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_feature_index.py tests/test_api.py tests/test_matcher.py -v`

Expected: tiny image build/cache/API tests pass.

### Task 4: Similarity Schema Bounds

**Files:**
- Modify: `src/crop_matcher/schemas.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `MatchItem.similarity` constrained inclusively to `0 <= value <= 100`, including generated OpenAPI schema.

- [ ] **Step 1: Write failing tests**

Validate that `-0.1` and `100.1` are rejected and OpenAPI declares minimum `0` and maximum `100`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_api.py -v`

Expected: invalid values currently validate.

- [ ] **Step 3: Add Pydantic field bounds**

Use `Field(ge=0, le=100)` on `MatchItem.similarity`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_api.py -v`

Expected: model and OpenAPI bounds pass.

### Task 5: Browser Success-Response Contract

**Files:**
- Modify: `src/crop_matcher/static/app.js`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `isValidMatchResponse(body) -> boolean`, checked before `renderQuery` can revoke/create an object URL or mutate result DOM.

- [ ] **Step 1: Write failing contract test**

Add a Node-backed browser-stub test that submits malformed successful JSON, expects `匹配失败，请重试`, and asserts zero `URL.createObjectURL`/`URL.revokeObjectURL` calls and no retained query URL.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_api.py -v`

Expected: malformed body reaches `renderQuery` and creates an object URL or throws after allocation.

- [ ] **Step 3: Validate complete response shape**

Require a positive integer query width/height, nonnegative integer elapsed time, a nonempty match array, and correctly typed/bounded fields for every match, including a safe image URL, before any rendering.

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_api.py -v`

Expected: malformed valid JSON displays the generic error without URL allocation.

### Task 6: Shared SIFT Concurrency

**Files:**
- Modify: `src/crop_matcher/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Produces: serialized calls to the shared `_sift.detectAndCompute`, independent from the existing FLANN lock.

- [ ] **Step 1: Write failing concurrency test**

Run two synchronized `match` calls against an overlap-detecting fake SIFT implementation and assert its maximum active call count is one.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_matcher.py -v`

Expected: maximum active SIFT calls is two.

- [ ] **Step 3: Add SIFT lock**

Initialize `_sift_lock = Lock()` and guard only `detectAndCompute`, retaining concurrency for the rest of matching.

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_matcher.py -v`

Expected: concurrent matches pass with maximum active SIFT calls of one.

### Task 7: Verification, Review, Report, And Single Commit

**Files:**
- Create outside worktree: `D:\workspace\auto-guessc\.git\worktrees\image-crop-matcher\sdd\final-fix-report.md`

**Interfaces:**
- Produces: one reviewed commit and an exact evidence report.

- [ ] **Step 1: Run required verification**

Run `pytest -v`, `ruff check .`, `ruff format --check .`, `node --check src/crop_matcher/static/app.js`, and `git diff --check`. Do not run the full benchmark.

- [ ] **Step 2: Self-review**

Inspect `git diff`, containment call sites, exception mappings, cache identity/invariants, frontend mutation ordering, and lock scope. Re-run affected checks after any correction.

- [ ] **Step 3: Create the coherent commit**

Inspect `git status`, `git diff`, and recent log; stage only intended worktree files and commit once with `fix: close final image matcher review findings`. Do not amend.

- [ ] **Step 4: Write final report**

Record status, changed files, each finding-to-test/fix mapping, exact command outcomes, commit hash, self-review, and concerns at the required administrative path.
