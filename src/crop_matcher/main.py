import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time

from fastapi import FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from crop_matcher.config import Settings
from crop_matcher.gallery_manager import GalleryConflictError, GalleryManager, GalleryPathError
from crop_matcher.gallery_state import GallerySelectionStore
from crop_matcher.imaging import ImageDecodeError, ImageTooLargeError, decode_image_bytes
from crop_matcher.schemas import (
    GalleryRequest,
    MatchItem,
    MatchResponse,
    QueryInfo,
    StatusResponse,
)

logger = logging.getLogger(__name__)


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
    manager = GalleryManager(
        resolved_settings,
        GallerySelectionStore(resolved_settings.selection_file),
    )
    build_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        initial_task = asyncio.create_task(asyncio.to_thread(manager.initialize))
        try:
            yield
        finally:
            await initial_task
            if build_tasks:
                await asyncio.gather(*build_tasks)

    app = FastAPI(lifespan=lifespan)
    app.state.gallery_manager = manager
    app.state.services = manager

    def status_response() -> StatusResponse:
        snapshot = manager.snapshot()
        active = snapshot.active
        return StatusResponse(
            state=snapshot.state,
            indexed_images=len(active.catalog.records) if active is not None else 0,
            build_time_ms=active.build_time_ms if active is not None else None,
            error=snapshot.initial_error,
            gallery_dir=str(active.gallery_dir) if active is not None else None,
            pending_gallery_dir=(
                str(snapshot.pending_gallery_dir)
                if snapshot.pending_gallery_dir is not None
                else None
            ),
            reindexing=snapshot.reindexing,
            switch_error=snapshot.switch_error,
        )

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
        return status_response()

    @app.post("/api/gallery", response_model=StatusResponse)
    async def switch_gallery(request: GalleryRequest, response: Response) -> StatusResponse:
        try:
            reservation = manager.reserve_switch(Path(request.path))
        except GalleryPathError:
            raise ApiError(400, "invalid_gallery", "The gallery path is invalid") from None
        except GalleryConflictError:
            raise ApiError(
                409,
                "gallery_switch_in_progress",
                "A gallery switch is already in progress",
            ) from None
        if reservation == "accepted":
            reserved_path = manager.snapshot().pending_gallery_dir
            if reserved_path is None:
                raise RuntimeError("Accepted gallery switch has no reservation")
            task = asyncio.create_task(
                asyncio.to_thread(manager.run_reserved_switch, reserved_path)
            )
            build_tasks.add(task)
            task.add_done_callback(build_tasks.discard)
            response.status_code = 202
        return status_response()

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
        snapshot = manager.snapshot()
        active = snapshot.active
        if active is None:
            raise ApiError(503, "service_unavailable", "The image index is not ready")

        started = time.perf_counter()
        try:
            results = await asyncio.to_thread(active.matcher.match_many, query, limit=5)
        except Exception:
            logger.exception("Image matching failed")
            raise ApiError(500, "internal_error", "An internal error occurred") from None
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return MatchResponse(
            query=QueryInfo(width=query.shape[1], height=query.shape[0]),
            elapsed_ms=elapsed_ms,
            matches=[
                MatchItem(
                    image_id=result.record.image_id,
                    parent_name=result.record.parent_name,
                    filename=result.record.filename,
                    width=result.record.width,
                    height=result.record.height,
                    similarity=result.similarity,
                    image_url=(f"/api/images/{active.cache_dir.name}/{result.record.image_id}"),
                )
                for result in results
            ],
        )

    @app.get("/api/images/{gallery_id}/{image_id}")
    async def get_image(gallery_id: str, image_id: str) -> FileResponse:
        snapshot = manager.snapshot()
        if snapshot.active is None:
            raise ApiError(503, "service_unavailable", "The image index is not ready")
        try:
            catalog = manager.image_catalog(gallery_id)
            record = catalog.get(image_id)
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
