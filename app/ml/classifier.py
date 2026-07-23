from pathlib import Path
from typing import Optional

import joblib
import numpy as np


class ClassifierHead:
    """Nearest-centroid classifier over frozen, L2-normalized CLIP embeddings:
    predicts via cosine similarity to each class's mean embedding, turned into
    pseudo-probabilities with a softmax.

    This is deliberately NOT a discriminatively-fit model (e.g. logistic
    regression). A linear classifier fit directly in a ~512-dim embedding
    space on the handful of examples a user grades in one sitting reaches
    100% training accuracy almost immediately while learning a decision
    boundary that fails to generalize at all past the exact training points
    (empirically ~0% accuracy on fresh same-class examples in testing,
    regardless of regularization strength) — there just isn't enough data to
    outweigh the dimensionality. Nearest-centroid sidesteps this: it only
    needs the mean direction of each class, which a handful of examples
    already estimates reasonably well, and it's the standard approach for
    few-shot classification on top of frozen embeddings.

    Each label (including good/great separately) gets its own centroid and
    all labels compete directly via a flat softmax — NOT hierarchically
    (not_part-vs-pooled-positive, then good-vs-great). A hierarchical version
    was tried and measured worse: taking the best-of-N per-class similarities
    (what flat argmax does) beats comparing against those classes' averaged/
    pooled centroid in the large majority of cases (~86% in a randomized
    check), since max(a, b) >= average(a, b) far more often than not. That
    directly trades away recall on the positive classes — pooling good+great
    to stabilize the not_part boundary sounds like it should help a sparse
    "great" class, but it costs exactly the "don't lose real positives"
    property this app's users care about most, so it isn't worth it."""

    # Controls how sharply cosine-similarity differences translate into
    # confidence gaps. Lower = more decisive/confident predictions. Chosen
    # empirically for CLIP's typical image-image cosine similarity range
    # (related images commonly land around 0.5-0.9, rarely near the extremes).
    TEMPERATURE = 0.1

    def __init__(self) -> None:
        self.centroids: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: list[str]) -> bool:
        """Returns False (no-op, leaves any existing centroids untouched) if
        fewer than 2 distinct classes are present."""
        labels = set(y)
        if len(labels) < 2:
            return False
        y_arr = np.asarray(y)
        centroids = {}
        for label in labels:
            mean = X[y_arr == label].mean(axis=0)
            norm = np.linalg.norm(mean)
            centroids[label] = mean / norm if norm > 0 else mean
        self.centroids = centroids
        return True

    def predict_proba(self, embedding: np.ndarray) -> Optional[dict[str, float]]:
        if not self.centroids:
            return None
        labels = list(self.centroids.keys())
        sims = np.array([embedding @ self.centroids[label] for label in labels])
        scaled = (sims - sims.max()) / self.TEMPERATURE
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return {label: float(p) for label, p in zip(labels, probs)}

    def predict_label(self, embedding: np.ndarray) -> Optional[str]:
        if not self.centroids:
            return None
        return max(self.centroids, key=lambda label: embedding @ self.centroids[label])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.centroids, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            centroids = joblib.load(path)
        except Exception:
            # Covers both a corrupted file and one saved by a previous,
            # differently-shaped classifier implementation.
            return False
        # Checking dict-ness alone isn't enough — a file saved by a
        # differently-shaped format can still be a dict (e.g. a hierarchical
        # variant tried and reverted during development saved
        # {"centroids": {...}, "positive_centroid": ...}), just not one
        # whose values are actually centroid vectors.
        if not isinstance(centroids, dict) or not all(isinstance(v, np.ndarray) for v in centroids.values()):
            return False
        self.centroids = centroids
        return True
