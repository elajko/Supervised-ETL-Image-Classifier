from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from app.config import SCORE_MAX, SCORE_MIN


class ScoreRegressor:
    """Predicts a continuous 0-100 score from a frozen CLIP embedding via
    Partial Least Squares regression.

    Chosen empirically over Ridge regression, k-NN regression, and a
    weighted-centroid-projection approach (the regression analog of the
    nearest-centroid classifier this replaced) via repeated held-out
    train/test splits on real graded data: PLS (n_components=10) had the
    best combination of Pearson correlation (rank-ordering, ~0.48, tied
    with the best Ridge result), MAE (~29 points on the 0-100 scale, best
    of all candidates), and bucket-agreement rate (~53%, notably ahead of
    every other candidate including Ridge's ~38%). PLS is built for exactly
    this regime -- high-dimensional features, limited samples, continuous
    target -- because it finds a small number of latent directions that
    jointly explain variance in both the embeddings and the score, unlike
    Ridge (which still fits all 512 dimensions directly, just regularized)
    or k-NN (which has no way to down-weight uninformative dimensions).

    These numbers are honest, not flattering. ~29 points of average error
    and ~53% bucket agreement reflects a genuinely hard problem, not a
    strong model -- the target concept turned out to be a matter of degree/
    emphasis within a visual genre, which general-purpose CLIP embeddings
    don't capture especially well (see the classifier comparison that
    preceded this, and the per-image misclassification investigation that
    motivated moving to a continuous score in the first place). PLS is
    simply the best of the options tested on this data, not a fix for that
    underlying limitation."""

    N_COMPONENTS = 10

    def __init__(self) -> None:
        self.model: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: list[float]) -> bool:
        """Returns False (no-op, leaves any existing model untouched) if
        fewer than 2 examples are available -- PLS needs at least 2 samples
        to fit at all."""
        if len(y) < 2:
            return False
        n_components = min(self.N_COMPONENTS, X.shape[0] - 1, X.shape[1])
        model = PLSRegression(n_components=n_components)
        model.fit(X, np.asarray(y, dtype=float))
        self.model = model
        return True

    def predict_score(self, embedding: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        raw = np.ravel(self.model.predict(embedding.reshape(1, -1)))[0]
        return float(np.clip(raw, SCORE_MIN, SCORE_MAX))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            model = joblib.load(path)
        except Exception:
            # Covers both a corrupted file and one saved by a previous,
            # differently-shaped model implementation (e.g. the old
            # ClassifierHead's centroid dict).
            return False
        if not isinstance(model, PLSRegression):
            return False
        self.model = model
        return True
