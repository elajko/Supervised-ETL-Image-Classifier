from pathlib import Path

from app.config import IMAGES_DIR, MODELS_DIR


def label_dir(session_id: str, label: str) -> Path:
    return IMAGES_DIR / session_id / label


def image_file_path(session_id: str, label: str, content_hash: str) -> Path:
    return label_dir(session_id, label) / f"{content_hash}.jpg"


def model_path(session_id: str) -> Path:
    return MODELS_DIR / session_id / "head.joblib"
