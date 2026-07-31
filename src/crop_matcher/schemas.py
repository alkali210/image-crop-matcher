from typing import Literal

from pydantic import BaseModel, Field


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


class QueryInfo(BaseModel):
    width: int
    height: int


class MatchItem(BaseModel):
    image_id: str
    parent_name: str
    filename: str
    width: int
    height: int
    similarity: float = Field(ge=0, le=100)
    image_url: str


class MatchResponse(BaseModel):
    query: QueryInfo
    elapsed_ms: int
    matches: list[MatchItem]
