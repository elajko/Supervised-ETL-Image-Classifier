import io
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED


@lru_cache(maxsize=1)
def _load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    return model, preprocess


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
