from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    gallery_dir: Path = Path("gallery")
    cache_dir: Path = Path(".cache")
    selection_file: Path = Path(".crop-matcher.json")
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    working_max_edge: int = 512
    query_feature_min_edge: int = 256
    sift_features: int = 1_000
    sift_contrast_threshold: float = 0.02
    candidate_count: int = 10
    tile_sizes: tuple[int, ...] = (64, 96, 128, 192, 256)
    coarse_template_edge: int = 16
