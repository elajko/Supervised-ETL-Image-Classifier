from app.storage.paths import image_file_path

DISCARDED_LABELS = {"not_part", "auto_not_part"}


def persist_graded_image(session_id: str, content_hash: str, image_bytes: bytes, label: str) -> str:
    """Write image bytes to disk under the given label's folder. Must not be called
    for a discarded label (not_part / auto_not_part) — those never touch disk."""
    if label in DISCARDED_LABELS:
        raise ValueError(f"label {label!r} is discarded and must not be persisted to disk")

    path = image_file_path(session_id, label, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return str(path)
