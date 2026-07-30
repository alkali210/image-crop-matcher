from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from crop_matcher.config import Settings
from crop_matcher.main import create_app


def encode(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, payload = cv2.imencode(extension, image)
    assert ok
    return payload.tobytes()


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
        unknown = client.get("/api/images/not-a-catalog-id")
        assert unknown.status_code == 404
        assert unknown.json() == {
            "error": {"code": "image_not_found", "message": "Image not found"}
        }
        assert client.get("/api/images/../../outside").status_code in {404, 422}


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
        services = client.app.state.services
        matcher = services.snapshot().matcher
        assert matcher is not None
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
    assert calls == [matcher.match]


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


def test_internal_match_error_is_generic_and_does_not_leak_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with make_client(tmp_path) as client:
        assert wait_until_ready(client)["state"] == "ready"
        matcher = client.app.state.services.snapshot().matcher
        assert matcher is not None

        def fail_match(_query: np.ndarray) -> None:
            raise RuntimeError(r"failed at D:\private\songs\secret.jpg")

        monkeypatch.setattr(matcher, "match", fail_match)
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
