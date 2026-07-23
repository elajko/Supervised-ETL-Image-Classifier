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
# A single page can easily reference hundreds of <img> tags (e.g. a forum/
# imageboard page's thread-list sidebar navigation). Processing them one at a
# time -- a real HTTP fetch each, before even reaching real content further
# down the page -- can make it look like the crawler has stalled entirely.
# Set fairly high since this is I/O-bound waiting, not CPU work: some sites
# serve a subset of images slowly/unreliably (observed: legacy thumbnails
# behind a CDN cold cache taking ~5s each to fail), and higher concurrency is
# what keeps those from serializing the whole page behind them.
MAX_CONCURRENT_IMAGE_FETCHES = 16
# CLIP embedding is CPU/memory-heavy (a full model forward pass), unlike the
# lightweight I/O-bound fetch above -- letting every image that passes the
# fetch+size filter run its embedding concurrently and unbounded (as opposed
# to just bounding the fetch step) can thrash a memory-constrained, CPU-only
# machine badly enough to make the crawler look completely stalled.
MAX_CONCURRENT_EMBEDDINGS = 2
PENDING_QUEUE_MAXSIZE = 5
NEXT_IMAGE_LONG_POLL_TIMEOUT = 10.0

# Listing/gallery pages commonly link full-resolution images via a small
# thumbnail <img> tag. Filtering by pixel dimensions (checked before the
# expensive CLIP embedding step) keeps thumbnails and UI icons out of the
# grading queue so the user only ever grades substantive images.
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

LABELS = ("not_part", "good", "great")
KEPT_LABELS = {"good", "great"}

# By design, not_part/auto_not_part images are normally discarded after their
# embedding is extracted (only the embedding + label row is kept, to save
# disk). Flip this on to also write those images to disk like any other
# label -- useful for building a held-out set of confirmed negative examples
# for offline evaluation, since none would otherwise exist on disk to test
# with.
SAVE_NOT_PART_IMAGES = True

# Sized generously so a model can be crawled into repeatedly over its
# lifetime without the bloom filter's false-positive rate degrading much
# past this target (it degrades gracefully beyond capacity, never causing
# false negatives — just falling back to the sorted-list check more often).
BLOOM_FILTER_CAPACITY = 50_000
BLOOM_FILTER_FALSE_POSITIVE_RATE = 0.01
