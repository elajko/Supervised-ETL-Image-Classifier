from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class ClassifierHead:
    """Thin wrapper around a LogisticRegression head trained on frozen CLIP
    embeddings. Always fully refit from scratch on all cached (embedding, label)
    pairs — at few-shot scale this is single-digit milliseconds, so there's no
    need for partial_fit's added complexity."""

    def __init__(self) -> None:
        self.model: Optional[LogisticRegression] = None

    def fit(self, X: np.ndarray, y: list[str]) -> bool:
        """Returns False (no-op, leaves any existing model untouched) if fewer
        than 2 distinct classes are present — LogisticRegression requires at
        least 2 classes to fit."""
        if len(set(y)) < 2:
            return False
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        self.model = clf
        return True

    def predict_proba(self, embedding: np.ndarray) -> Optional[dict[str, float]]:
        if self.model is None:
            return None
        probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
        return {str(label): float(p) for label, p in zip(self.model.classes_, probs)}

    def predict_label(self, embedding: np.ndarray) -> Optional[str]:
        probs = self.predict_proba(embedding)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        self.model = joblib.load(path)
        return True
