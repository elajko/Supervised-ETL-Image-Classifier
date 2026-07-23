from app.config import SAVE_NOT_PART_IMAGES
from app.storage.paths import image_file_path

NOT_PART_LABELS = {"not_part", "auto_not_part"}


def should_persist(label: str) -> bool:
    """Whether a graded/auto-filed image with this label should be written
    to disk. Every label is persisted except not_part/auto_not_part, which
    are normally discarded -- unless SAVE_NOT_PART_IMAGES opts back in."""
    if label in NOT_PART_LABELS:
        return SAVE_NOT_PART_IMAGES
    return True


def persist_graded_image(session_id: str, content_hash: str, image_bytes: bytes, label: str) -> str:
    """Write image bytes to disk under the given label's folder. Callers must
    check should_persist(label) first -- this raises if asked to persist a
    label that's currently configured to be discarded."""
    if not should_persist(label):
        raise ValueError(f"label {label!r} is discarded and must not be persisted to disk")

    path = image_file_path(session_id, label, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return str(path)
