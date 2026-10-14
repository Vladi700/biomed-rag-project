"""JSON I/O and archive helpers used by the notebook."""

import json
import shutil
from pathlib import Path


class StaleArtifactError(RuntimeError):
    """A saved artifact no longer matches the artifacts it was built from."""


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_zip(src_dir: Path, dest_dir: Path | None = None) -> Path:
    """Zip `src_dir` into `<dest_dir>/<name>.zip`, with the folder itself as the archive root."""
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir) if dest_dir else src_dir.parent
    archive = shutil.make_archive(
        str(dest_dir / src_dir.name), "zip", root_dir=src_dir.parent, base_dir=src_dir.name
    )
    return Path(archive)
