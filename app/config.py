from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "app.db"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = DATA_DIR / "models"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_MODEL_ID = f"{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}"

MAX_DEPTH = 1
MAX_PAGES_PER_SESSION = 200
MAX_CONCURRENT_PAGES = 4
PENDING_QUEUE_MAXSIZE = 5
NEXT_IMAGE_LONG_POLL_TIMEOUT = 10.0

LABELS = ("not_part", "part", "textbook")
KEPT_LABELS = {"part", "textbook"}
AUTO_LABEL_PREFIX = "auto_"
