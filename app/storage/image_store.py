import shutil
from pathlib import Path
from typing import Optional

from app.storage.paths import image_file_path


def persist_graded_image(session_id: str, content_hash: str, image_bytes: bytes, label: str) -> str:
    """Write image bytes to disk under the given label's (bucket) folder."""
    path = image_file_path(session_id, label, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return str(path)


def move_graded_image(session_id: str, content_hash: str, old_path: Optional[Path], new_label: str) -> str:
    """Moves an already-persisted image file to its new label's (bucket)
    folder -- used when promoting an auto-filed image to a human-confirmed
    score/bucket. Returns the new local_path."""
    new_path = image_file_path(session_id, new_label, content_hash)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    if old_path and old_path != new_path and old_path.exists():
        shutil.move(str(old_path), str(new_path))
    return str(new_path)
