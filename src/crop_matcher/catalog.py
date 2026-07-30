from dataclasses import dataclass, replace
from hashlib import blake2s
from pathlib import Path

from crop_matcher.imaging import ImageDecodeError, ImageTooLargeError, read_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class CatalogManifestEntry:
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    path: Path
    relative_path: Path
    parent_name: str
    filename: str
    width: int
    height: int


class ImageCatalog:
    def __init__(
        self,
        root: Path,
        records: tuple[ImageRecord, ...],
        manifest: tuple[CatalogManifestEntry, ...],
    ) -> None:
        self.root = root.resolve()
        self.records = records
        self.manifest = manifest
        self._by_id = {record.image_id: record for record in records}

    @classmethod
    def scan(cls, root: Path, max_pixels: int) -> "ImageCatalog":
        root = root.resolve()
        records: list[ImageRecord] = []
        manifest: list[CatalogManifestEntry] = []
        paths = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not path.stem.lower().endswith("_256")
            ),
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
        for path in paths:
            relative = path.relative_to(root)
            try:
                resolved_path = cls._resolve_contained_path(root, path)
                image = read_image(resolved_path, max_pixels)
                stat = resolved_path.stat()
            except (ImageDecodeError, ImageTooLargeError, KeyError, OSError):
                continue
            normalized = relative.as_posix()
            records.append(
                ImageRecord(
                    image_id=blake2s(normalized.encode("utf-8"), digest_size=12).hexdigest(),
                    path=path,
                    relative_path=relative,
                    parent_name=relative.parent.name,
                    filename=relative.name,
                    width=image.shape[1],
                    height=image.shape[0],
                )
            )
            manifest.append(CatalogManifestEntry(normalized, stat.st_size, stat.st_mtime_ns))
        return cls(root, tuple(records), tuple(manifest))

    def get(self, image_id: str) -> ImageRecord:
        record = self._by_id[image_id]
        return replace(record, path=self._resolve_contained_path(self.root, record.path))

    @staticmethod
    def _resolve_contained_path(root: Path, path: Path) -> Path:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise KeyError(path)
        except (OSError, ValueError) as exc:
            raise KeyError(path) from exc
        return resolved
