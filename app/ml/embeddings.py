import io
import os
import threading

# The CLIP checkpoint is already fully downloaded/cached locally (see
# open_clip's HF Hub cache) after first run, so there's no need to hit the
# network on every startup just to check for updates -- that's all the
# "unauthenticated requests" rate-limit warning was about. Offline mode skips
# that check entirely. Uses setdefault so an explicit HF_HUB_OFFLINE in the
# environment (e.g. to force a re-check) still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_preprocess = None
_load_lock = threading.Lock()


def _load_model():
    # embed_image() runs on a thread pool (asyncio.to_thread), so more than
    # one real OS thread can call this concurrently. A plain @lru_cache does
    # NOT guard against that -- two threads can both see a cache miss and
    # both start loading at once, and concurrently loading the model twice
    # (e.g. racing on open_clip/HF Hub's own on-disk cache locking) hangs
    # rather than erroring, verified empirically. Double-checked locking
    # ensures the actual load only ever happens once; everyone else blocks
    # on the lock instead of racing into the load themselves.
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess
    with _load_lock:
        if _model is not None:
            return _model, _preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        model.eval()
        _model, _preprocess = model, preprocess
        return _model, _preprocess


def embed_image(image_bytes: bytes) -> np.ndarray:
    """Compute a frozen CLIP embedding for raw image bytes. Returns a float32
    array of shape (embed_dim,), L2-normalized."""
    model, preprocess = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).numpy().astype(np.float32)


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
