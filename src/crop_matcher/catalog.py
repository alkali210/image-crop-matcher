from dataclasses import asdict, dataclass, replace
from hashlib import blake2s, sha256
import json
import logging
import os
from pathlib import Path
import stat
import tempfile

from crop_matcher.imaging import ImageDecodeError, ImageTooLargeError, read_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CATALOG_CACHE_SCHEMA_VERSION = 3
logger = logging.getLogger(__name__)


class _CatalogChangedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogManifestEntry:
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _CatalogSourceEntry:
    relative_path: str
    lexical_size: int | None
    lexical_mtime_ns: int | None
    resolved_identity: str | None
    resolved_relative_path: str | None
    size: int | None
    mtime_ns: int | None


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
        for _attempt in range(3):
            sources = cls._discover_sources(root)
            try:
                catalog = cls._scan_sources(root, sources, max_pixels)
            except _CatalogChangedError:
                continue
            if cls._discover_sources(root) == sources:
                return catalog
        raise _CatalogChangedError("Gallery changed repeatedly while scanning")

    @classmethod
    def load_or_scan(cls, root: Path, max_pixels: int, cache_path: Path) -> "ImageCatalog":
        root = root.resolve()
        for _attempt in range(3):
            sources = cls._discover_sources(root)
            expected = {
                "schema_version": CATALOG_CACHE_SCHEMA_VERSION,
                "gallery_root": os.path.normcase(str(root)),
                "max_pixels": max_pixels,
                "sources": [asdict(source) for source in sources],
            }
            cached = cls._load_cache(cache_path, root, sources, max_pixels, expected)
            if cached is not None:
                return cached
            try:
                catalog = cls._scan_sources(root, sources, max_pixels)
            except _CatalogChangedError:
                continue
            if cls._discover_sources(root) != sources:
                continue
            body = {
                **expected,
                "records": [
                    {
                        "image_id": record.image_id,
                        "relative_path": record.relative_path.as_posix(),
                        "width": record.width,
                        "height": record.height,
                    }
                    for record in catalog.records
                ],
                "manifest": [asdict(entry) for entry in catalog.manifest],
            }
            payload = {**body, "cache_identity": cls._cache_identity(body)}
            cls._save_cache(cache_path, payload)
            return catalog
        raise _CatalogChangedError("Gallery changed repeatedly while caching")

    @classmethod
    def _load_cache(
        cls,
        cache_path: Path,
        root: Path,
        sources: tuple[_CatalogSourceEntry, ...],
        max_pixels: int,
        expected: dict[str, object],
    ) -> "ImageCatalog | None":
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text("utf-8"), parse_constant=cls._reject_constant)
            if not isinstance(payload, dict):
                return None
            if {key: payload[key] for key in expected} != expected:
                return None
            identity = payload.pop("cache_identity")
            if not isinstance(identity, str) or identity != cls._cache_identity(payload):
                return None
            return cls._from_cache(root, payload, sources, max_pixels)
        except (KeyError, OSError, OverflowError, RecursionError, TypeError, ValueError):
            return None

    @classmethod
    def _discover_sources(cls, root: Path) -> tuple[_CatalogSourceEntry, ...]:
        discovered: list[_CatalogSourceEntry] = []
        for path in cls._candidate_paths(root):
            relative = path.relative_to(root)
            lexical_size: int | None = None
            lexical_mtime_ns: int | None = None
            resolved_identity: str | None = None
            source_size: int | None = None
            source_mtime_ns: int | None = None
            try:
                lexical_stat = path.lstat()
                lexical_size = lexical_stat.st_size
                lexical_mtime_ns = lexical_stat.st_mtime_ns
                resolved_path = path.resolve(strict=True)
                source_stat = resolved_path.stat()
                resolved_identity = cls._path_identity(resolved_path)
                source_size = source_stat.st_size
                source_mtime_ns = source_stat.st_mtime_ns
                if not stat.S_ISREG(source_stat.st_mode):
                    raise KeyError(path)
                resolved_relative_path = resolved_path.relative_to(root).as_posix()
            except (KeyError, OSError, ValueError) as exc:
                logger.warning("Skipping source image %s: %s", relative.as_posix(), exc)
                discovered.append(
                    _CatalogSourceEntry(
                        relative_path=relative.as_posix(),
                        lexical_size=lexical_size,
                        lexical_mtime_ns=lexical_mtime_ns,
                        resolved_identity=resolved_identity,
                        resolved_relative_path=None,
                        size=source_size,
                        mtime_ns=source_mtime_ns,
                    )
                )
            else:
                discovered.append(
                    _CatalogSourceEntry(
                        relative_path=relative.as_posix(),
                        lexical_size=lexical_size,
                        lexical_mtime_ns=lexical_mtime_ns,
                        resolved_identity=resolved_identity,
                        resolved_relative_path=resolved_relative_path,
                        size=source_size,
                        mtime_ns=source_mtime_ns,
                    )
                )
        return tuple(discovered)

    @classmethod
    def _scan_sources(
        cls, root: Path, sources: tuple[_CatalogSourceEntry, ...], max_pixels: int
    ) -> "ImageCatalog":
        records: list[ImageRecord] = []
        manifest: list[CatalogManifestEntry] = []
        for source in sources:
            if source.resolved_relative_path is None:
                continue
            relative = Path(source.relative_path)
            path = root / relative
            try:
                if cls._describe_source(root, path) != source:
                    raise _CatalogChangedError(path)
                resolved_path = cls._resolve_contained_path(root, path)
                image = read_image(resolved_path, max_pixels)
                if cls._describe_source(root, path) != source:
                    raise _CatalogChangedError(path)
            except _CatalogChangedError:
                raise
            except (ImageDecodeError, ImageTooLargeError, KeyError, OSError) as exc:
                logger.warning("Skipping source image %s: %s", relative.as_posix(), exc)
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
            if source.size is None or source.mtime_ns is None:
                raise ValueError("Valid source metadata is incomplete")
            manifest.append(CatalogManifestEntry(normalized, source.size, source.mtime_ns))
        return cls(root, tuple(records), tuple(manifest))

    @classmethod
    def _describe_source(cls, root: Path, path: Path) -> _CatalogSourceEntry:
        relative = path.relative_to(root)
        lexical_stat = path.lstat()
        resolved_path = cls._resolve_contained_path(root, path)
        source_stat = resolved_path.stat()
        return _CatalogSourceEntry(
            relative_path=relative.as_posix(),
            lexical_size=lexical_stat.st_size,
            lexical_mtime_ns=lexical_stat.st_mtime_ns,
            resolved_identity=cls._path_identity(resolved_path),
            resolved_relative_path=resolved_path.relative_to(root).as_posix(),
            size=source_stat.st_size,
            mtime_ns=source_stat.st_mtime_ns,
        )

    @staticmethod
    def _candidate_paths(root: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in root.rglob("*")
                if path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not path.stem.lower().endswith("_256")
            ),
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )

    @classmethod
    def _from_cache(
        cls,
        root: Path,
        payload: dict[str, object],
        sources: tuple[_CatalogSourceEntry, ...],
        max_pixels: int,
    ) -> "ImageCatalog":
        records_data = payload["records"]
        manifest_data = payload["manifest"]
        if not isinstance(records_data, list) or not isinstance(manifest_data, list):
            raise ValueError("Invalid catalog cache")
        valid_sources = {
            source.relative_path: source
            for source in sources
            if source.resolved_relative_path is not None
        }
        records: list[ImageRecord] = []
        record_paths: list[str] = []
        for item in records_data:
            if not isinstance(item, dict):
                raise ValueError("Invalid catalog record")
            relative = Path(str(item["relative_path"]))
            normalized = relative.as_posix()
            if relative.is_absolute() or ".." in relative.parts or normalized not in valid_sources:
                raise ValueError("Invalid catalog record path")
            if normalized in record_paths:
                raise ValueError("Duplicate catalog record")
            image_id = blake2s(normalized.encode("utf-8"), digest_size=12).hexdigest()
            if item["image_id"] != image_id:
                raise ValueError("Invalid catalog image ID")
            width = cls._require_json_int(item["width"])
            height = cls._require_json_int(item["height"])
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError("Invalid cached image dimensions")
            record_paths.append(normalized)
            records.append(
                ImageRecord(
                    image_id=image_id,
                    path=root / relative,
                    relative_path=relative,
                    parent_name=relative.parent.name,
                    filename=relative.name,
                    width=width,
                    height=height,
                )
            )
        source_order = [source.relative_path for source in sources]
        if record_paths != sorted(record_paths, key=source_order.index):
            raise ValueError("Invalid catalog record order")
        manifest: list[CatalogManifestEntry] = []
        for record_path, item in zip(record_paths, manifest_data, strict=True):
            if not isinstance(item, dict) or item.get("relative_path") != record_path:
                raise ValueError("Invalid catalog manifest")
            source = valid_sources[record_path]
            size = cls._require_json_int(item.get("size"))
            mtime_ns = cls._require_json_int(item.get("mtime_ns"))
            if size != source.size or mtime_ns != source.mtime_ns:
                raise ValueError("Stale catalog manifest")
            manifest.append(CatalogManifestEntry(record_path, size, mtime_ns))
        if len(manifest) != len(records):
            raise ValueError("Invalid catalog manifest")
        return cls(root, tuple(records), tuple(manifest))

    @staticmethod
    def _cache_identity(payload: dict[str, object]) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    @staticmethod
    def _require_json_int(value: object) -> int:
        if type(value) is not int:
            raise ValueError("Expected a JSON integer")
        return value

    @staticmethod
    def _path_identity(path: Path) -> str:
        normalized = os.path.normcase(str(path))
        return sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _save_cache(cache_path: Path, payload: dict[str, object]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(payload, temporary, ensure_ascii=False, separators=(",", ":"))
                temporary_path = Path(temporary.name)
            temporary_path.replace(cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get(self, image_id: str) -> ImageRecord:
        record = self._by_id[image_id]
        return replace(record, path=self._resolve_contained_path(self.root, record.path))

    @staticmethod
    def _resolve_contained_path(root: Path, path: Path) -> Path:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if not stat.S_ISREG(resolved.stat().st_mode):
                raise KeyError(path)
        except (OSError, ValueError) as exc:
            raise KeyError(f"{path}: {exc}") from exc
        return resolved
