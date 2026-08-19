from dataclasses import replace
from io import BytesIO
from pathlib import Path
import subprocess
from threading import Event, Thread
import time
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from crop_matcher.config import Settings
from crop_matcher.main import create_app
from crop_matcher.matcher import MatchResult
from crop_matcher.schemas import MatchItem


def encode(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, payload = cv2.imencode(extension, image)
    assert ok
    return payload.tobytes()


def compressed_png(width: int, height: int) -> bytes:
    payload = BytesIO()
    Image.new("1", (width, height)).save(payload, format="PNG")
    return payload.getvalue()


def make_client(tmp_path: Path, **setting_overrides: Any) -> TestClient:
    gallery = tmp_path / "songs"
    image = np.zeros((220, 220, 3), np.uint8)
    cv2.putText(image, "API", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    path = gallery / "api-song" / "base.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(encode(image, ".jpg"))
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
        **setting_overrides,
    )
    return TestClient(create_app(settings))


def wait_until_ready(client: TestClient) -> dict[str, object]:
    for _ in range(200):
        status = client.get("/api/status").json()
        if status["state"] != "building":
            return status
        time.sleep(0.01)
    pytest.fail("catalog did not finish building")


def match_item(similarity: float) -> MatchItem:
    return MatchItem(
        image_id="id",
        parent_name="parent",
        filename="image.png",
        width=10,
        height=10,
        similarity=similarity,
        image_url="/api/images/id",
    )


@pytest.mark.parametrize("similarity", [-0.1, 100.1])
def test_match_item_rejects_similarity_outside_percentage_range(similarity: float) -> None:
    with pytest.raises(ValidationError):
        match_item(similarity)


def test_match_item_similarity_bounds_are_inclusive_and_in_openapi(tmp_path: Path) -> None:
    assert match_item(0).similarity == 0
    assert match_item(100).similarity == 100

    schema = create_app(
        Settings(gallery_dir=tmp_path / "songs", cache_dir=tmp_path / "cache")
    ).openapi()["components"]["schemas"]["MatchItem"]["properties"]["similarity"]

    assert schema["minimum"] == 0
    assert schema["maximum"] == 100


def test_status_match_and_safe_image_delivery(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        initial = client.get("/api/status")
        assert initial.status_code == 200
        assert initial.json()["state"] in {"building", "ready"}
        status = wait_until_ready(client)
        assert status["state"] == "ready"
        assert status["indexed_images"] == 1
        assert isinstance(status["build_time_ms"], int)
        assert status["build_time_ms"] >= 0
        assert status["error"] is None
        assert status["gallery_dir"] == str((tmp_path / "songs").resolve())
        assert status["pending_gallery_dir"] is None
        assert status["reindexing"] is False
        assert status["switch_error"] is None

        source = cv2.imread(str(tmp_path / "songs" / "api-song" / "base.jpg"))
        query = cv2.resize(source[30:190, 30:190], (90, 90))
        response = client.post(
            "/api/match",
            files={"file": ("query.png", encode(query), "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == {"width": 90, "height": 90}
        assert body["elapsed_ms"] >= 0
        assert len(body["matches"]) == 1
        assert body["matches"][0]["parent_name"] == "api-song"
        assert body["matches"][0]["filename"] == "base.jpg"
        assert body["matches"][0]["width"] == 220
        assert body["matches"][0]["height"] == 220
        assert 0 <= body["matches"][0]["similarity"] <= 100
        image_response = client.get(body["matches"][0]["image_url"])
        assert image_response.status_code == 200
        assert image_response.headers["content-type"].startswith("image/")
        gallery_image_prefix = body["matches"][0]["image_url"].rsplit("/", 1)[0]
        unknown = client.get(f"{gallery_image_prefix}/not-a-catalog-id")
        assert unknown.status_code == 404
        assert unknown.json() == {
            "error": {"code": "image_not_found", "message": "Image not found"}
        }
        assert client.get(f"{gallery_image_prefix}/../../outside").status_code in {404, 422}


def test_match_image_url_keeps_gallery_identity_across_atomic_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    relative = Path("same") / "base.png"
    first_payload = encode(np.full((32, 32, 3), 40, np.uint8))
    second_payload = encode(np.full((32, 32, 3), 220, np.uint8))
    for gallery, payload in ((first, first_payload), (second, second_payload)):
        path = gallery / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    settings = Settings(
        gallery_dir=first,
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
    )
    with TestClient(create_app(settings)) as client:
        assert wait_until_ready(client)["state"] == "ready"
        manager = client.app.state.gallery_manager
        active = manager.snapshot().active
        assert active is not None
        match_started = Event()
        match_released = Event()

        def blocked_match(_query: np.ndarray, limit: int) -> list[MatchResult]:
            assert limit == 5
            match_started.set()
            assert match_released.wait(timeout=5)
            record = active.catalog.records[0]
            return [MatchResult(record, 90.0, "template", 0, 0.0, 0.9)]

        monkeypatch.setattr(active.matcher, "match_many", blocked_match)
        responses: list[Any] = []

        def request_match() -> None:
            responses.append(
                client.post(
                    "/api/match",
                    files={
                        "file": (
                            "query.png",
                            encode(np.zeros((8, 8, 3), np.uint8)),
                            "image/png",
                        )
                    },
                )
            )

        worker = Thread(target=request_match)
        worker.start()
        assert match_started.wait(timeout=5)
        assert manager.reserve_switch(second) == "accepted"
        manager.run_reserved_switch(second)
        match_released.set()
        worker.join(timeout=5)
        assert not worker.is_alive()

        response = responses[0]
        assert response.status_code == 200
        image_url = response.json()["matches"][0]["image_url"]
        assert image_url.count("/") == 4
        image_response = client.get(image_url)

    assert image_response.status_code == 200
    assert image_response.content == first_payload
    assert image_response.content != second_payload


def test_match_returns_five_ranked_distinct_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    for seed in range(5):
        image = np.full((32, 32, 3), seed * 50, np.uint8)
        path = gallery / f"song-{seed}" / "base.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(encode(image))
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")

    with TestClient(create_app(settings)) as client:
        assert wait_until_ready(client)["state"] == "ready"
        active = client.app.state.gallery_manager.snapshot().active
        assert active is not None
        by_parent = {record.parent_name: record for record in active.catalog.records}
        ranked = [
            MatchResult(by_parent["song-2"], 90.0, "sift", 4, 1.0, 0.9),
            MatchResult(by_parent["song-0"], 80.0, "template", 0, 0.0, 0.8),
            MatchResult(by_parent["song-1"], 70.0, "template", 0, 0.0, 0.7),
            MatchResult(by_parent["song-3"], 60.0, "template", 0, 0.0, 0.6),
            MatchResult(by_parent["song-4"], 50.0, "template", 0, 0.0, 0.5),
        ]
        monkeypatch.setattr(active.matcher, "match_many", lambda _query, limit: ranked[:limit])

        response = client.post(
            "/api/match",
            files={"file": ("query.png", encode(np.zeros((8, 8, 3), np.uint8)), "image/png")},
        )

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 5
    assert [match["parent_name"] for match in matches] == [
        "song-2",
        "song-0",
        "song-1",
        "song-3",
        "song-4",
    ]
    assert len({match["image_id"] for match in matches}) == 5
    assert [match["similarity"] for match in matches] == [90.0, 80.0, 70.0, 60.0, 50.0]


def test_image_delivery_returns_404_for_catalog_path_outside_gallery(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert wait_until_ready(client)["state"] == "ready"
        active = client.app.state.gallery_manager.snapshot().active
        assert active is not None
        catalog = active.catalog
        record = catalog.records[0]
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(record.path.read_bytes())
        catalog._by_id[record.image_id] = replace(record, path=outside)

        response = client.get(f"/api/images/{active.cache_dir.name}/{record.image_id}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "image_not_found"


def test_image_delivery_is_unavailable_during_initial_build(tmp_path: Path) -> None:
    (tmp_path / "songs").mkdir()
    app = create_app(
        Settings(
            gallery_dir=tmp_path / "songs",
            cache_dir=tmp_path / "cache",
            selection_file=tmp_path / "selection.json",
        )
    )
    manager = app.state.gallery_manager
    started = Event()
    released = Event()

    def blocking_builder(_gallery_dir: Path, _cache_dir: Path) -> None:
        started.set()
        assert released.wait(timeout=5)
        raise RuntimeError("expected build stop")

    manager._builder = blocking_builder
    try:
        with TestClient(app) as client:
            assert started.wait(timeout=5)
            response = client.get("/api/images/gallery/image")
            assert response.status_code == 503
            assert response.json()["error"] == {
                "code": "service_unavailable",
                "message": "The image index is not ready",
            }
            released.set()
    finally:
        released.set()


def test_image_delivery_is_unavailable_after_initial_build_error(tmp_path: Path) -> None:
    settings = Settings(
        gallery_dir=tmp_path / "missing",
        cache_dir=tmp_path / "cache",
        selection_file=tmp_path / "selection.json",
    )
    with TestClient(create_app(settings)) as client:
        assert wait_until_ready(client)["state"] == "error"
        response = client.get("/api/images/gallery/image")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": "The image index is not ready",
    }


def test_rejects_invalid_oversized_and_excessive_pixel_uploads(tmp_path: Path) -> None:
    with make_client(tmp_path, max_upload_bytes=10_000, max_image_pixels=50_000) as client:
        assert wait_until_ready(client)["state"] == "ready"

        invalid = client.post(
            "/api/match",
            files={"file": ("bad.jpg", b"bad", "image/jpeg")},
        )
        oversized = client.post(
            "/api/match",
            files={"file": ("large.bin", b"x" * 10_001, "application/octet-stream")},
        )
        excessive_pixels = client.post(
            "/api/match",
            files={
                "file": (
                    "large.png",
                    encode(np.zeros((225, 225, 3), np.uint8)),
                    "image/png",
                )
            },
        )

        assert invalid.status_code == 400
        assert invalid.json() == {
            "error": {
                "code": "invalid_image",
                "message": "The uploaded file is not a supported image",
            }
        }
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "file_too_large"
        assert excessive_pixels.status_code == 413
        assert excessive_pixels.json()["error"] == {
            "code": "image_too_large",
            "message": "Decoded image exceeds the pixel limit",
        }


def test_rejects_compressed_oversized_upload_before_opencv_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with make_client(tmp_path, max_image_pixels=25_000_000) as client:
        assert wait_until_ready(client)["state"] == "ready"

        def fail_decode(*_args: object) -> None:
            pytest.fail("cv2.imdecode must not receive an oversized upload")

        monkeypatch.setattr(cv2, "imdecode", fail_decode)
        response = client.post(
            "/api/match",
            files={
                "file": (
                    "compressed.png",
                    compressed_png(5_001, 5_000),
                    "image/png",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_upload_reads_only_limit_plus_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_sizes: list[int] = []
    original_read = UploadFile.read

    async def recording_read(upload: UploadFile, size: int = -1) -> bytes:
        requested_sizes.append(size)
        return await original_read(upload, size)

    monkeypatch.setattr(UploadFile, "read", recording_read)

    with make_client(tmp_path, max_upload_bytes=32) as client:
        response = client.post(
            "/api/match",
            files={"file": ("large.bin", b"x" * 100, "application/octet-stream")},
        )

    assert response.status_code == 413
    assert requested_sizes == [33]


def test_matching_runs_in_a_worker_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import crop_matcher.main as main_module

    with make_client(tmp_path) as client:
        assert wait_until_ready(client)["state"] == "ready"
        active = client.app.state.gallery_manager.snapshot().active
        assert active is not None
        matcher = active.matcher
        calls: list[object] = []
        original_to_thread = main_module.asyncio.to_thread

        async def recording_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
            calls.append(function)
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(main_module.asyncio, "to_thread", recording_to_thread)
        source = cv2.imread(str(tmp_path / "songs" / "api-song" / "base.jpg"))
        response = client.post(
            "/api/match",
            files={"file": ("query.png", encode(source[30:190, 30:190]), "image/png")},
        )

    assert response.status_code == 200
    assert calls == [matcher.match_many]


def test_build_failure_and_unavailable_match_are_structured(tmp_path: Path) -> None:
    settings = Settings(gallery_dir=tmp_path / "missing", cache_dir=tmp_path / "cache")

    with TestClient(create_app(settings)) as client:
        status = wait_until_ready(client)
        response = client.post(
            "/api/match",
            files={"file": ("query.png", encode(np.zeros((8, 8, 3), np.uint8)), "image/png")},
        )

    assert status["state"] == "error"
    assert status["indexed_images"] == 0
    assert status["error"] == "No supported images found"
    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": "The image index is not ready",
    }


def test_gallery_api_validates_conflicts_and_accepts_switch(tmp_path: Path) -> None:
    second = tmp_path / "second"
    second.mkdir()
    with make_client(tmp_path) as client:
        assert wait_until_ready(client)["state"] == "ready"

        relative = client.post("/api/gallery", json={"path": "relative"})
        missing = client.post("/api/gallery", json={"path": str(tmp_path / "missing")})
        same = client.post("/api/gallery", json={"path": str(tmp_path / "songs")})
        manager = client.app.state.gallery_manager
        assert manager.reserve_switch(second) == "accepted"
        pending = client.post("/api/gallery", json={"path": str(tmp_path / "songs")})
        manager.run_reserved_switch(second)

        third = tmp_path / "third"
        third.mkdir()
        accepted = client.post("/api/gallery", json={"path": str(third)})

    assert relative.status_code == 400
    assert relative.json()["error"]["code"] == "invalid_gallery"
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "invalid_gallery"
    assert same.status_code == 200
    assert pending.status_code == 409
    assert pending.json()["error"]["code"] == "gallery_switch_in_progress"
    assert accepted.status_code == 202


def test_shutdown_waits_for_accepted_gallery_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crop_matcher.main as main_module

    client = make_client(tmp_path)
    client.__enter__()
    manager = client.app.state.gallery_manager
    started = Event()
    released = Event()
    shutdown_wait_started = Event()
    shutdown_finished = Event()
    second = tmp_path / "second"
    second.mkdir()

    def blocking_builder(_gallery_dir: Path, _cache_dir: Path) -> None:
        started.set()
        assert released.wait(timeout=5)
        raise RuntimeError("expected test build failure")

    assert wait_until_ready(client)["state"] == "ready"
    manager._builder = blocking_builder
    response = client.post("/api/gallery", json={"path": str(second)})
    assert response.status_code == 202
    assert started.wait(timeout=5)
    original_gather = main_module.asyncio.gather

    async def recording_gather(*awaitables: object) -> list[object]:
        shutdown_wait_started.set()
        return await original_gather(*awaitables)

    monkeypatch.setattr(main_module.asyncio, "gather", recording_gather)

    def shut_down() -> None:
        client.__exit__(None, None, None)
        shutdown_finished.set()

    worker = Thread(target=shut_down)
    worker.start()
    try:
        assert shutdown_wait_started.wait(timeout=5)
        assert not shutdown_finished.is_set()
    finally:
        released.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert shutdown_finished.is_set()


def test_tiny_featureless_gallery_becomes_ready_and_matches(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    source_path = gallery / "tiny" / "base.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(encode(np.zeros((20, 30, 3), np.uint8)))
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        tile_sizes=(64, 96),
    )

    with TestClient(create_app(settings)) as client:
        status = wait_until_ready(client)
        response = client.post(
            "/api/match",
            files={
                "file": (
                    "query.png",
                    encode(np.zeros((10, 10, 3), np.uint8)),
                    "image/png",
                )
            },
        )

    assert status["state"] == "ready"
    assert status["indexed_images"] == 1
    assert response.status_code == 200
    assert response.json()["matches"][0]["parent_name"] == "tiny"


def test_internal_match_error_is_generic_and_does_not_leak_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with make_client(tmp_path) as client:
        assert wait_until_ready(client)["state"] == "ready"
        active = client.app.state.gallery_manager.snapshot().active
        assert active is not None
        matcher = active.matcher

        def fail_match(_query: np.ndarray, limit: int) -> None:
            assert limit == 5
            raise RuntimeError(r"failed at D:\private\songs\secret.jpg")

        monkeypatch.setattr(matcher, "match_many", fail_match)
        response = client.post(
            "/api/match",
            files={"file": ("query.png", encode(np.zeros((8, 8, 3), np.uint8)), "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An internal error occurred"}
    }
    assert "private" not in response.text
    assert "secret.jpg" not in response.text


def test_missing_upload_and_static_hooks_do_not_require_static_files(tmp_path: Path) -> None:
    app = create_app(Settings(gallery_dir=tmp_path / "songs", cache_dir=tmp_path / "cache"))
    route_paths = {route.path for route in app.routes}

    assert "/" in route_paths
    assert "/static" in route_paths
    with TestClient(app) as client:
        response = client.post("/api/match")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_root_serves_functional_static_shell(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="drop-zone"' in response.text
        assert 'id="result-list"' in response.text
        assert '<label class="visually-hidden" for="file-input">' in response.text
        assert 'class="brand" href="/" aria-label=' not in response.text
        assert '<link rel="icon" href="data:,">' in response.text
        assert "Find the frame" not in response.text
        assert "几何内点稳定" not in response.text
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200


def test_webui_keeps_upload_controls_beside_results(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        html = client.get("/").text

    assert '<main id="search-workspace" class="search-workspace">' in html
    assert 'id="upload-view"' in html
    assert 'id="results-view"' in html
    assert 'id="results-empty"' in html
    assert 'id="reupload-button"' not in html


def test_webui_contains_runtime_gallery_controls(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        html = client.get("/").text

    assert '<label for="gallery-path">' in html
    assert 'id="gallery-path"' in html
    assert 'autocomplete="off"' in html
    assert 'id="current-gallery-path"' in html
    assert 'id="switch-gallery-button"' in html
    assert 'id="gallery-switch-status"' in html
    assert 'aria-live="polite"' in html


def test_frontend_submits_gallery_path_and_keeps_upload_during_reindex(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        script = client.get("/static/app.js").text

    assert 'fetch("/api/gallery"' in script
    assert "pending_gallery_dir" in script
    assert "reindexing" in script
    assert "galleryPending" in script
    assert "switchGalleryButton.disabled = pending" in script
    assert "status.gallery_dir" in script
    assert "status.switch_error" in script


def test_frontend_tracks_active_availability_across_poll_and_switch_races() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor() {
    this.attrs = new Map();
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.hidden = true;
    this.listeners = {};
    this.textContent = "";
    this.value = "";
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append() {}
  click() {}
  contains() { return false; }
  focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); }
  replaceChildren() {}
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}

const selectors = [
  "#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status",
  "#error-message", "#upload-view", "#results-view", "#results-heading", "#query-summary",
  "#query-preview", "#query-name", "#query-meta", "#result-list", "#results-empty",
  "#current-gallery-path", "#gallery-switch-form", "#gallery-path",
  "#switch-gallery-button", "#gallery-switch-status",
];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#drop-zone").setAttribute("aria-disabled", "true");
elements.get("#loading-status").hidden = true;

const responses = [];
let timerCallback = null;
const windowObject = {
  addEventListener() {},
  clearTimeout() { timerCallback = null; },
  location: { origin: "http://testserver" },
  setTimeout(callback) { timerCallback = callback; return 1; },
};
const context = {
  console,
  document: {
    createElement() { return new Element(); },
    createTextNode(text) { return { textContent: text }; },
    querySelector(selector) { return elements.get(selector); },
  },
  fetch: (...args) => {
    if (responses.length === 0) throw new Error(`unexpected fetch: ${args[0]}`);
    return responses.shift()(...args);
  },
  FormData: class { append() {} },
  URL,
  window: windowObject,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);

const jsonResponse = (body, ok = true) => async () => ({ ok, json: async () => body });
const flush = async () => {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
};
const assert = (condition, message) => { if (!condition) throw new Error(message); };

(async () => {
  responses.push(jsonResponse({
    state: "building", indexed_images: 0, build_time_ms: null, error: null, gallery_dir: null,
    pending_gallery_dir: null, reindexing: false, switch_error: null,
  }));
  await context.pollStatus();
  assert(elements.get("#file-input").disabled, "initial build enabled upload");
  assert(timerCallback !== null, "initial build did not schedule polling");

  responses.push(async () => { throw new Error("status unavailable"); });
  const retryAfterBuild = timerCallback;
  retryAfterBuild();
  await flush();
  assert(elements.get("#file-input").disabled, "failed follow-up enabled upload without active bundle");
  assert(timerCallback !== null, "status failure did not schedule retry");

  responses.push(jsonResponse({
    state: "ready", indexed_images: 3, build_time_ms: 4, error: null, gallery_dir: "C:\\active",
    pending_gallery_dir: "C:\\pending", reindexing: true, switch_error: null,
  }));
  await context.pollStatus();
  assert(!elements.get("#file-input").disabled, "active reindex disabled upload");
  assert(elements.get("#switch-gallery-button").disabled, "reindex did not disable switch");
  assert(timerCallback !== null, "reindex did not keep polling");

  let resolveSubmission;
  responses.push(() => new Promise((resolve) => { resolveSubmission = resolve; }));
  elements.get("#gallery-path").value = "  C:\\next  ";
  const submission = elements.get("#gallery-switch-form").listeners.submit({ preventDefault() {} });
  await flush();
  assert(elements.get("#switch-gallery-button").disabled, "submission did not disable switch");

  responses.push(jsonResponse({
    state: "ready", indexed_images: 3, build_time_ms: 4, error: null, gallery_dir: "C:\\active",
    pending_gallery_dir: null, reindexing: false, switch_error: null,
  }));
  await context.pollStatus();
  assert(elements.get("#switch-gallery-button").disabled, "poll won submission race");
  resolveSubmission({
    ok: true,
    json: async () => ({
      state: "ready", indexed_images: 3, build_time_ms: 4, error: null, gallery_dir: "C:\\active",
      pending_gallery_dir: "C:\\next", reindexing: true, switch_error: null,
    }),
  });
  await submission;
  assert(elements.get("#switch-gallery-button").disabled, "accepted switch was not pending");
  assert(!elements.get("#file-input").disabled, "accepted switch disabled active upload");

  responses.push(jsonResponse({
    state: "ready", indexed_images: 3, build_time_ms: 4, error: null, gallery_dir: "C:\\active",
    pending_gallery_dir: null, reindexing: false, switch_error: "No supported images found",
  }));
  await context.pollStatus();
  assert(!elements.get("#file-input").disabled, "switch failure disabled active upload");
  assert(!elements.get("#switch-gallery-button").disabled, "switch failure left switch disabled");
  assert(
    elements.get("#gallery-switch-status").textContent.includes("No supported images found"),
    "switch failure was not rendered",
  );
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""

    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_frontend_ignores_poll_started_before_accepted_gallery_reservation() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs");
const vm = require("vm");
class Element {
  constructor() { this.attrs = new Map(); this.dataset = {}; this.disabled = false; this.files = []; this.hidden = true; this.listeners = {}; this.textContent = ""; this.value = ""; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append() {} click() {} contains() { return false; } focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); } replaceChildren() {}
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}
const selectors = ["#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status", "#error-message", "#upload-view", "#results-view", "#results-heading", "#query-summary", "#query-preview", "#query-name", "#query-meta", "#result-list", "#results-empty", "#current-gallery-path", "#gallery-switch-form", "#gallery-path", "#switch-gallery-button", "#gallery-switch-status"];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#drop-zone").setAttribute("aria-disabled", "false");
elements.get("#loading-status").hidden = true;
let timerCallback = null;
let resolveOldPoll;
const responses = [
  () => new Promise((resolve) => { resolveOldPoll = resolve; }),
  async () => ({ ok: true, json: async () => ({ state: "ready", indexed_images: 1, build_time_ms: 4, error: null, gallery_dir: "C:\\old", pending_gallery_dir: "C:\\new", reindexing: true, switch_error: null }) }),
];
const context = {
  console,
  document: { createElement() { return new Element(); }, createTextNode(text) { return { textContent: text }; }, querySelector(selector) { return elements.get(selector); } },
  fetch: (...args) => responses.shift()(...args), FormData: class { append() {} }, URL,
  window: { addEventListener() {}, clearTimeout() { timerCallback = null; }, location: { origin: "http://testserver" }, setTimeout(callback) { timerCallback = callback; return 1; } },
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
const flush = async () => { await new Promise((resolve) => setImmediate(resolve)); await new Promise((resolve) => setImmediate(resolve)); };
const assert = (condition, message) => { if (!condition) throw new Error(message); };
(async () => {
  const oldPoll = context.pollStatus();
  elements.get("#gallery-path").value = "C:\\new";
  const submission = elements.get("#gallery-switch-form").listeners.submit({ preventDefault() {} });
  await submission;
  resolveOldPoll({ ok: true, json: async () => ({ state: "ready", indexed_images: 1, build_time_ms: 4, error: null, gallery_dir: "C:\\old", pending_gallery_dir: null, reindexing: false, switch_error: null }) });
  await oldPoll; await flush();
  assert(elements.get("#switch-gallery-button").disabled, "stale pre-reservation poll enabled switching");
  assert(timerCallback !== null, "stale pre-reservation poll stopped accepted-switch polling");
})().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
"""
    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_frontend_reconciles_ambiguous_gallery_submission_failure() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs"); const vm = require("vm");
class Element {
  constructor() { this.attrs = new Map(); this.dataset = {}; this.disabled = false; this.files = []; this.hidden = true; this.listeners = {}; this.textContent = ""; this.value = ""; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append() {} click() {} contains() { return false; } focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); } replaceChildren() {}
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}
const selectors = ["#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status", "#error-message", "#upload-view", "#results-view", "#results-heading", "#query-summary", "#query-preview", "#query-name", "#query-meta", "#result-list", "#results-empty", "#current-gallery-path", "#gallery-switch-form", "#gallery-path", "#switch-gallery-button", "#gallery-switch-status"];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#loading-status").hidden = true; elements.get("#gallery-path").value = "C:\\new";
let timerCallback = null;
const responses = [
  async () => { throw new Error("response lost"); },
  async () => ({ ok: true, json: async () => ({ state: "ready", indexed_images: 1, build_time_ms: 4, error: null, gallery_dir: "C:\\old", pending_gallery_dir: "C:\\new", reindexing: true, switch_error: null }) }),
  async () => ({ ok: true, json: async () => ({ state: "ready", indexed_images: 2, build_time_ms: 5, error: null, gallery_dir: "C:\\new", pending_gallery_dir: null, reindexing: false, switch_error: null }) }),
];
const context = {
  console, Error, document: { createElement() { return new Element(); }, createTextNode(text) { return { textContent: text }; }, querySelector(selector) { return elements.get(selector); } },
  fetch: (...args) => responses.shift()(...args), FormData: class { append() {} }, URL,
  window: { addEventListener() {}, clearTimeout() { timerCallback = null; }, location: { origin: "http://testserver" }, setTimeout(callback) { timerCallback = () => { timerCallback = null; callback(); }; return 1; } },
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
(async () => {
  await elements.get("#gallery-switch-form").listeners.submit({ preventDefault() {} });
  assert(elements.get("#switch-gallery-button").disabled, "ambiguous failure enabled another submission");
  assert(timerCallback !== null, "ambiguous failure did not schedule reconciliation");
  assert(elements.get("#gallery-switch-status").textContent.includes("response lost"), "immediate error was lost");
  let poll = timerCallback; poll(); await new Promise((resolve) => setImmediate(resolve));
  assert(elements.get("#switch-gallery-button").disabled, "pending authoritative status enabled submission");
  assert(timerCallback !== null, "pending authoritative status stopped polling");
  poll = timerCallback; poll(); await new Promise((resolve) => setImmediate(resolve));
  assert(!elements.get("#switch-gallery-button").disabled, "completed authoritative status did not enable submission");
  assert(timerCallback === null, "completed authoritative status kept polling");
})().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
"""
    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_frontend_rejects_incomplete_status_for_submission_and_polling() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs"); const vm = require("vm");
class Element {
  constructor() { this.attrs = new Map(); this.dataset = {}; this.disabled = false; this.files = []; this.hidden = true; this.listeners = {}; this.textContent = ""; this.value = ""; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append() {} click() {} contains() { return false; } focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); } replaceChildren() {}
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}
const selectors = ["#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status", "#error-message", "#upload-view", "#results-view", "#results-heading", "#query-summary", "#query-preview", "#query-name", "#query-meta", "#result-list", "#results-empty", "#current-gallery-path", "#gallery-switch-form", "#gallery-path", "#switch-gallery-button", "#gallery-switch-status"];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#loading-status").hidden = true; elements.get("#gallery-path").value = "C:\\new";
let timerCallback = null;
const responses = [
  async () => ({ ok: true, json: async () => ({ state: "ready", indexed_images: 0, build_time_ms: 4, error: null, gallery_dir: "C:\\new", pending_gallery_dir: null, reindexing: false, switch_error: null }) }),
  async () => ({ ok: true, json: async () => ({ state: "error", indexed_images: 0, build_time_ms: null, error: "   ", gallery_dir: null, pending_gallery_dir: null, reindexing: false, switch_error: null }) }),
  async () => ({ ok: true, json: async () => ({ state: "ready", indexed_images: 2, build_time_ms: 4, error: null, gallery_dir: "C:\\new", pending_gallery_dir: null, reindexing: false, switch_error: null }) }),
  async () => ({ ok: true, json: async () => ({ reindexing: false }) }),
];
const context = {
  console, Error, document: { createElement() { return new Element(); }, createTextNode(text) { return { textContent: text }; }, querySelector(selector) { return elements.get(selector); } },
  fetch: (...args) => responses.shift()(...args), FormData: class { append() {} }, URL,
  window: { addEventListener() {}, clearTimeout() { timerCallback = null; }, location: { origin: "http://testserver" }, setTimeout(callback) { timerCallback = () => { timerCallback = null; callback(); }; return 1; } },
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
(async () => {
  assert(context.isValidStatusResponse({ state: "building", indexed_images: 0, build_time_ms: null, error: null, gallery_dir: null, pending_gallery_dir: null, reindexing: false, switch_error: null }), "legitimate building status was rejected");
  assert(context.isValidStatusResponse({ state: "error", indexed_images: 0, build_time_ms: null, error: "No supported images found", gallery_dir: null, pending_gallery_dir: null, reindexing: false, switch_error: null }), "legitimate error status was rejected");
  context.renderStatus({ state: "ready", indexed_images: 1, build_time_ms: 3, error: null, gallery_dir: "C:\\old", pending_gallery_dir: null, reindexing: false, switch_error: null });
  await elements.get("#gallery-switch-form").listeners.submit({ preventDefault() {} });
  assert(elements.get("#switch-gallery-button").disabled, "incomplete POST status enabled another submission");
  assert(timerCallback !== null, "incomplete POST status did not schedule reconciliation");
  assert(!elements.get("#file-input").disabled, "incomplete POST status disabled known active upload");
  let poll = timerCallback; poll(); await new Promise((resolve) => setImmediate(resolve));
  assert(elements.get("#switch-gallery-button").disabled, "invalid error status cleared ambiguous submission lock");
  assert(!elements.get("#file-input").disabled, "invalid error status replaced known active availability");
  assert(timerCallback !== null, "invalid error status did not schedule reconciliation");
  poll = timerCallback; poll(); await new Promise((resolve) => setImmediate(resolve));
  assert(!elements.get("#switch-gallery-button").disabled, "valid reconciliation did not unlock submission");
  assert(!elements.get("#file-input").disabled, "valid reconciliation disabled active upload");
  await context.pollStatus();
  assert(!elements.get("#file-input").disabled, "incomplete poll disabled known active upload");
  assert(timerCallback !== null, "incomplete poll did not schedule retry");
})().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
"""
    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_upload_shell_lists_exact_supported_formats_and_limits(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'accept=".jpg,.jpeg,.png,.webp,.bmp"' in response.text
    assert "JPG、JPEG、PNG、WebP、BMP" in response.text
    assert "10 MiB" in response.text
    assert "25 MP" in response.text


def test_frontend_script_contains_retry_lifecycle_and_error_contracts(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        script = client.get("/static/app.js")

    assert script.status_code == 200
    assert 'const STATUS_RETRY_MESSAGE = "暂时无法读取索引，正在重试...";' in script.text
    assert "function scheduleStatusPoll()" in script.text
    assert "function parseJsonResponse(response)" in script.text
    assert 'const MATCH_ERROR_MESSAGE = "匹配失败，请重试";' in script.text
    assert 'window.addEventListener("pagehide", (event) =>' in script.text
    assert "if (event.persisted)" in script.text
    assert 'window.addEventListener("pageshow", restorePage);' in script.text


def test_frontend_rejects_malformed_success_before_allocating_query_url() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor() {
    this.attrs = new Map();
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.hidden = true;
    this.listeners = {};
    this.textContent = "";
    this.value = "";
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append() {}
  click() {}
  contains() { return false; }
  focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); }
  replaceChildren() {}
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}

const selectors = [
  "#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status",
  "#error-message", "#upload-view", "#results-view", "#results-heading", "#query-summary",
  "#query-preview", "#query-name", "#query-meta", "#result-list", "#results-empty",
  "#current-gallery-path", "#gallery-switch-form", "#gallery-path",
  "#switch-gallery-button", "#gallery-switch-status",
];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#drop-zone").setAttribute("aria-disabled", "false");
elements.get("#index-status").dataset.state = "ready";
elements.get("#loading-status").hidden = true;
const file = { name: "query.png", size: 100 };
elements.get("#file-input").files = [file];

let createdUrls = 0;
let revokedUrls = 0;
const windowObject = {
  addEventListener() {},
  clearTimeout() {},
  location: { origin: "http://testserver" },
  setTimeout() { return 1; },
};
const context = {
  console,
  document: {
    createElement() { return new Element(); },
    createTextNode(text) { return { textContent: text }; },
    querySelector(selector) { return elements.get(selector); },
  },
  fetch: async () => ({
    ok: true,
    json: async () => ({ query: {}, elapsed_ms: 4, matches: [] }),
  }),
  FormData: class { append() {} },
  URL: {
    createObjectURL() { createdUrls += 1; return "blob:test"; },
    revokeObjectURL() { revokedUrls += 1; },
  },
  window: windowObject,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);

(async () => {
  elements.get("#file-input").listeners.change();
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  const error = elements.get("#error-message");
  if (error.hidden || error.textContent !== "匹配失败，请重试") {
    throw new Error(`generic error was not shown: ${error.textContent}`);
  }
  if (createdUrls !== 0 || revokedUrls !== 0) {
    throw new Error(`unexpected URL lifecycle: created=${createdUrls} revoked=${revokedUrls}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""

    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_frontend_replaces_results_after_direct_second_upload() -> None:
    app_script = Path(__file__).parents[1] / "src" / "crop_matcher" / "static" / "app.js"
    harness = r"""
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor() {
    this.attrs = new Map();
    this.children = [];
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.hidden = true;
    this.listeners = {};
    this.textContent = "";
    this.value = "";
    this.replaceCount = 0;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append(...children) { this.children.push(...children); }
  click() {}
  contains() { return false; }
  focus() {}
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); }
  replaceChildren(...children) {
    this.children = children;
    this.replaceCount += 1;
  }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
}

const selectors = [
  "#drop-zone", "#file-input", "#index-status", "#status-text", "#loading-status",
  "#error-message", "#results-heading", "#query-summary", "#query-preview", "#query-name",
  "#query-meta", "#result-list", "#results-empty", "#current-gallery-path",
  "#gallery-switch-form", "#gallery-path", "#switch-gallery-button",
  "#gallery-switch-status",
];
const elements = new Map(selectors.map((selector) => [selector, new Element()]));
elements.get("#drop-zone").setAttribute("aria-disabled", "true");
elements.get("#loading-status").hidden = true;
elements.get("#results-empty").hidden = false;

function match(imageId, parentName, rank) {
  return {
    image_id: imageId,
    parent_name: parentName,
    filename: `${imageId}.jpg`,
    width: 768,
    height: 768,
    similarity: 100 - rank,
    image_url: `/api/images/gallery/${imageId}`,
  };
}

function response(prefix, parentName) {
  return {
    query: { width: 384, height: 384 },
    elapsed_ms: 10,
    matches: Array.from(
      { length: 5 },
      (_value, index) => match(`${prefix}-${index + 1}`, parentName, index),
    ),
  };
}

const responses = [response("first", "first-parent"), response("second", "second-parent")];
let createdUrls = 0;
let revokedUrls = 0;
const NativeURL = URL;
class TestURL extends NativeURL {
  static createObjectURL() {
    createdUrls += 1;
    return `blob:query-${createdUrls}`;
  }
  static revokeObjectURL() { revokedUrls += 1; }
}
const windowObject = {
  addEventListener() {},
  clearTimeout() {},
  location: { origin: "http://testserver" },
  setTimeout() { return 1; },
};
const context = {
  console,
  document: {
    createElement() { return new Element(); },
    createTextNode(text) { return { textContent: text }; },
    querySelector(selector) { return elements.get(selector); },
  },
  fetch: async () => ({ ok: true, json: async () => responses.shift() }),
  FormData: class { append() {} },
  URL: TestURL,
  window: windowObject,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);

async function flushUpload() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

(async () => {
  context.renderStatus({
    state: "ready",
    indexed_images: 10,
    build_time_ms: 1,
    error: null,
    gallery_dir: "C:\\gallery",
    pending_gallery_dir: null,
    reindexing: false,
    switch_error: null,
  });

  const fileInput = elements.get("#file-input");
  const resultList = elements.get("#result-list");
  fileInput.files = [{ name: "first.png", size: 100 }];
  fileInput.listeners.change();
  await flushUpload();
  if (resultList.children.length !== 5 || resultList.replaceCount !== 1) {
    throw new Error("first upload did not render exactly five replacement rows");
  }

  fileInput.files = [{ name: "second.png", size: 200 }];
  fileInput.listeners.change();
  await flushUpload();
  if (resultList.children.length !== 5 || resultList.replaceCount !== 2) {
    throw new Error(
      `second upload appended stale results: rows=${resultList.children.length} replacements=${resultList.replaceCount}`,
    );
  }
  if (elements.get("#query-name").textContent !== "second.png") {
    throw new Error(`query summary was not replaced: ${elements.get("#query-name").textContent}`);
  }
  if (
    elements.get("#drop-zone").getAttribute("aria-disabled") !== "false" ||
    fileInput.disabled
  ) {
    throw new Error("uploader was not re-enabled after the second search");
  }
  if (createdUrls !== 2 || revokedUrls !== 1) {
    throw new Error(`unexpected URL lifecycle: created=${createdUrls} revoked=${revokedUrls}`);
  }
  if (!elements.get("#results-empty").hidden || elements.get("#query-summary").hidden) {
    throw new Error("results state did not replace the empty state");
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""

    subprocess.run(
        ["node", "-e", harness, str(app_script)],
        check=True,
        capture_output=True,
        text=True,
    )
