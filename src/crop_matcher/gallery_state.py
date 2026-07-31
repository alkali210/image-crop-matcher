from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile


class GallerySelectionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Path | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text("utf-8"))
            value = payload["gallery_dir"]
            if not isinstance(value, str) or not value:
                raise ValueError
            gallery = Path(value)
            if not gallery.is_absolute():
                raise ValueError
            return gallery.resolve(strict=False)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("Invalid gallery selection file") from exc

    def save(self, gallery_dir: Path) -> None:
        resolved = gallery_dir.resolve(strict=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                json.dump({"gallery_dir": str(resolved)}, temporary, ensure_ascii=False)
                temporary_path = Path(temporary.name)
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def gallery_cache_dir(cache_root: Path, gallery_dir: Path) -> Path:
    normalized = os.path.normcase(str(gallery_dir.resolve(strict=True)))
    digest = sha256(normalized.encode("utf-8")).hexdigest()
    return cache_root / "galleries" / digest
