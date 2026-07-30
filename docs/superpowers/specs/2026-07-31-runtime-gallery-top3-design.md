# Runtime Gallery Selection and Top-3 Results Design

## Summary

Extend the local crop matcher so a user can enter any absolute server-local image directory in the WebUI, build its index in the background, and remember the successful selection across restarts. Search remains available through the previous gallery until the new index is ready. Each query returns up to the three highest-confidence distinct source images.

## Goals

- Accept an absolute gallery path from the WebUI without uploading or copying the gallery.
- Persist only a successfully indexed path and restore it on the next startup.
- Keep the current matcher available while another gallery builds, then atomically swap all gallery services.
- Maintain independent catalog and feature caches per absolute gallery path.
- Return `min(3, gallery image count)` distinct results in descending similarity order.
- Preserve the current compact dark WebUI and existing result item fields.

## Non-Goals

- Browser directory upload through `webkitdirectory`.
- A Python desktop directory-picker dialog.
- Multiple simultaneous active galleries or merged cross-gallery search.
- Cancelling a build already in progress or queueing additional gallery changes.
- Editing, deleting, or otherwise managing source files through the WebUI.

## Persisted Selection and Cache Namespaces

The application stores the last successful absolute gallery path in `.crop-matcher.json`, which is local runtime state and is excluded from Git. The file is written through a temporary sibling followed by atomic replacement. A missing selection file falls back to the configured default `songs/` directory.

If the persisted directory no longer exists, startup enters `error` without silently changing the selection. The WebUI remains available so the user can submit a replacement directory.

Each gallery receives a cache namespace below `.cache/galleries/<digest>/`, where `<digest>` is the SHA-256 digest of the normalized, resolved absolute directory path. `catalog.json`, `manifest.json`, and `features.npz` remain unchanged within that namespace. Switching back to a previously indexed, unchanged gallery reuses its namespace.

## Gallery Manager

`GalleryManager` replaces direct startup ownership by `AppServices`. It holds one immutable active service bundle containing `ImageCatalog`, `FeatureIndex`, `ImageMatcher`, resolved gallery path, build duration, and cache namespace. Request handlers take a bundle snapshot, so an in-flight query keeps a valid reference even if a new bundle becomes active.

Only one replacement build may run at a time:

1. Validate that the submitted value is a nonempty absolute path to an existing directory.
2. Resolve the path and reject a second switch with `409` while one is pending.
3. Build the catalog, index, and matcher in a worker thread using the path-specific cache namespace.
4. Continue serving searches and images from the old bundle during the build.
5. On success, atomically persist the path; only after persistence succeeds, replace the complete active bundle.
6. On failure, retain the old bundle, clear the pending path, and expose a concise switch error.

If there is no active bundle during initial startup, state is `building`, `ready`, or `error` as today. When an active bundle exists, a replacement build keeps state `ready` and reports `reindexing=true`.

## API Changes

### `GET /api/status`

Adds:

```json
{
  "gallery_dir": "D:\\images\\gallery",
  "pending_gallery_dir": null,
  "reindexing": false,
  "switch_error": null
}
```

`gallery_dir` is the active resolved absolute path or `null` when no bundle is active. `pending_gallery_dir` is populated only during a replacement build.

### `POST /api/gallery`

Request:

```json
{"path": "D:\\images\\gallery"}
```

- Returns `202` with the updated status when a build is accepted.
- Returns `200` when the resolved path is already active and no build is needed.
- Returns `400` for empty, relative, missing, or non-directory paths.
- Returns `409` while another replacement build is pending.
- A directory with no supported source images is a build failure, not a successful empty gallery.

The endpoint never accepts a browser file list and never exposes arbitrary files; original-image delivery remains restricted to opaque IDs in the active catalog.

## Top-3 Matching

`ImageMatcher.match_many(query_bgr, limit=3)` becomes the list-producing interface. Existing `match(query_bgr)` returns the first item from `match_many(query_bgr, limit=1)` for benchmark and caller compatibility.

The matcher collects geometrically verified SIFT candidates. If fewer than `limit` distinct candidates are available, it evaluates pHash/template candidates and fills missing slots. Candidate image IDs are deduplicated before ranking. Because fallback retrieval already deterministically fills from the indexed image set, the final count is `min(limit, gallery image count)`.

Similarity remains method-aware:

- SIFT candidates retain the existing geometry/appearance base score. The winner-versus-runner margin bonus is computed once and applied equally to every SIFT candidate, preserving their base ordering and the current first-result score.
- pHash candidates retain their appearance-based score and 89.9 cap. Their margin bonus is also computed once and applied equally to all pHash candidates.
- The merged set is sorted by final similarity descending, then by image ID for deterministic ties.

`POST /api/match` calls `match_many(..., limit=3)` and serializes every result into the existing `matches` array. No response field is removed or renamed.

## WebUI

The upload page adds a restrained gallery-directory block immediately below index status:

- A read-only current-directory line.
- An absolute-path text input.
- A `切换图库` button.
- A local status/error line for background switching.

The upload control stays enabled while `reindexing=true`. The directory button is disabled during a pending build. On success, the displayed active path and indexed-image count update. On failure, the error appears in the directory block and the old gallery remains usable.

The existing result list renders up to three rows and requires no new visual component. Rank labels become `匹配 #1`, `匹配 #2`, and `匹配 #3` in API order.

## Error and Concurrency Handling

- Directory path validation occurs before a build task is created.
- Persistence failure prevents activation so memory state and remembered state cannot disagree.
- Search and image requests use one active-bundle snapshot per request.
- A successful swap changes catalog, index, matcher, path, and status together.
- Replacement failure never clears an existing active bundle.
- Shutdown awaits any initial or replacement build task.
- API errors remain structured and do not expose stack traces.

## Testing

- Selection store atomic round-trip, missing file, malformed file, and successful-path-only persistence.
- Cache namespace stability and separation for two absolute galleries.
- Startup from persisted path and default fallback when no selection exists.
- Invalid path validation, no-image failure, same-path no-op, and concurrent-switch `409`.
- Search remains available against the old matcher during replacement build.
- Successful atomic swap and failed-build rollback.
- `match_many` ordering, deduplication, SIFT plus fallback filling, limit handling, deterministic ties, and galleries containing one or two images.
- API response contains exactly three results for galleries of at least three images.
- WebUI path submission, switching state, failure recovery, Top-3 rows, keyboard behavior, desktop layout, and mobile layout.
