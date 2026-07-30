import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Literal

import cv2
from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import ImageDecodeError, ImageTooLargeError, decode_image_bytes
from crop_matcher.matcher import ImageMatcher
from crop_matcher.schemas import MatchItem, MatchResponse, QueryInfo, StatusResponse

logger = logging.getLogger(__name__)

ServiceState = Literal["building", "ready", "error"]


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    state: ServiceState
    error: str | None
    build_time_ms: int | None
    catalog: ImageCatalog | None
    feature_index: FeatureIndex | None
    matcher: ImageMatcher | None


@dataclass(slots=True)
class AppServices:
    state: ServiceState = "building"
    error: str | None = None
    build_time_ms: int | None = None
    catalog: ImageCatalog | None = None
    feature_index: FeatureIndex | None = None
    matcher: ImageMatcher | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                self.state,
                self.error,
                self.build_time_ms,
                self.catalog,
                self.feature_index,
                self.matcher,
            )

    def set_ready(
        self,
        catalog: ImageCatalog,
        feature_index: FeatureIndex,
        matcher: ImageMatcher,
        build_time_ms: int,
    ) -> None:
        with self._lock:
            self.catalog = catalog
            self.feature_index = feature_index
            self.matcher = matcher
            self.build_time_ms = build_time_ms
            self.error = None
            self.state = "ready"

    def set_error(self, message: str, build_time_ms: int) -> None:
        with self._lock:
            self.catalog = None
            self.feature_index = None
            self.matcher = None
            self.build_time_ms = build_time_ms
            self.error = message
            self.state = "error"


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    services = AppServices()

    def build_services() -> None:
        started = time.perf_counter()
        try:
            catalog = ImageCatalog.scan(
                resolved_settings.gallery_dir,
                resolved_settings.max_image_pixels,
            )
            if not catalog.records:
                services.set_error(
                    "No supported images found",
                    round((time.perf_counter() - started) * 1000),
                )
                return
            feature_index = FeatureIndex.load_or_build(catalog, resolved_settings)
            matcher = ImageMatcher(catalog, feature_index, resolved_settings)
        except Exception:
            logger.exception("Failed to initialize the image catalog and feature index")
            services.set_error(
                "Failed to initialize image index",
                round((time.perf_counter() - started) * 1000),
            )
            return
        services.set_ready(
            catalog,
            feature_index,
            matcher,
            round((time.perf_counter() - started) * 1000),
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        build_task = asyncio.create_task(asyncio.to_thread(build_services))
        try:
            yield
        finally:
            await build_task

    app = FastAPI(lifespan=lifespan)
    app.state.services = services

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(422, "invalid_request", "The request is invalid")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        messages = {
            404: ("not_found", "Not found"),
            405: ("method_not_allowed", "Method not allowed"),
        }
        code, message = messages.get(exc.status_code, ("request_error", "The request failed"))
        return error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def handle_internal_error(_request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error")
        return error_response(500, "internal_error", "An internal error occurred")

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        snapshot = services.snapshot()
        return StatusResponse(
            state=snapshot.state,
            indexed_images=len(snapshot.catalog.records) if snapshot.catalog is not None else 0,
            build_time_ms=snapshot.build_time_ms,
            error=snapshot.error,
        )

    @app.post("/api/match", response_model=MatchResponse)
    async def match_image(file: UploadFile) -> MatchResponse:
        payload = await file.read(resolved_settings.max_upload_bytes + 1)
        if len(payload) > resolved_settings.max_upload_bytes:
            raise ApiError(413, "file_too_large", "The uploaded file exceeds the size limit")
        try:
            query = decode_image_bytes(payload, resolved_settings.max_image_pixels)
        except ImageDecodeError as exc:
            raise ApiError(400, "invalid_image", str(exc)) from None
        except ImageTooLargeError as exc:
            raise ApiError(413, "image_too_large", str(exc)) from None
        except cv2.error:
            raise ApiError(
                400,
                "invalid_image",
                "The uploaded file is not a supported image",
            ) from None

        snapshot = services.snapshot()
        if snapshot.state != "ready" or snapshot.matcher is None:
            raise ApiError(503, "service_unavailable", "The image index is not ready")

        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(snapshot.matcher.match, query)
        except Exception:
            logger.exception("Image matching failed")
            raise ApiError(500, "internal_error", "An internal error occurred") from None
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        record = result.record
        return MatchResponse(
            query=QueryInfo(width=query.shape[1], height=query.shape[0]),
            elapsed_ms=elapsed_ms,
            matches=[
                MatchItem(
                    image_id=record.image_id,
                    parent_name=record.parent_name,
                    filename=record.filename,
                    width=record.width,
                    height=record.height,
                    similarity=result.similarity,
                    image_url=f"/api/images/{record.image_id}",
                )
            ],
        )

    @app.get("/api/images/{image_id}")
    async def get_image(image_id: str) -> FileResponse:
        snapshot = services.snapshot()
        if snapshot.state != "ready" or snapshot.catalog is None:
            raise ApiError(503, "service_unavailable", "The image index is not ready")
        try:
            record = snapshot.catalog.get(image_id)
        except KeyError:
            raise ApiError(404, "image_not_found", "Image not found") from None
        return FileResponse(
            record.path,
            filename=record.filename,
            content_disposition_type="inline",
        )

    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    async def get_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
