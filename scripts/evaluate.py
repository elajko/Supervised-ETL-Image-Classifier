"""Evaluate the regressor against a model's own human-graded images via
repeated held-out train/test splits.

Only human-graded scores (bucket low/medium/high) are used -- auto-filed
(auto-low/auto-medium/auto-high) images are the regressor's own past
predictions, not verified ground truth, so they can't be used to measure
accuracy.

Usage:
    .venv/bin/python3 scripts/evaluate.py <model name or id substring> [--splits 20] [--test-frac 0.3]
"""

import argparse
import asyncio
import sys
from collections import Counter

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, ".")

from app.db import HUMAN_BUCKETS, get_training_data, list_sessions  # noqa: E402
from app.labels import score_to_bucket  # noqa: E402
from app.ml.embeddings import deserialize_embedding  # noqa: E402
from app.ml.regressor import ScoreRegressor  # noqa: E402


async def resolve_session(query: str) -> tuple[str, str]:
    sessions = await list_sessions()
    matches = [s for s in sessions if query.lower() in s["name"].lower() or query == s["id"]]
    if not matches:
        raise SystemExit(f"no model matching {query!r}. Available: {[s['name'] for s in sessions]}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous match for {query!r}: {[s['name'] for s in matches]}")
    return matches[0]["id"], matches[0]["name"]


def stratified_split(y: np.ndarray, test_frac: float, rng: np.random.Generator, n_bins: int = 5) -> tuple:
    """Splits indices so the score range is represented proportionally in
    both train and test -- a plain random split can otherwise skew one side
    toward a narrower score range."""
    bins = np.clip((y / 100 * n_bins).astype(int), 0, n_bins - 1)
    train_idx, test_idx = [], []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def evaluate_once(X: np.ndarray, y: np.ndarray, test_frac: float, rng: np.random.Generator) -> dict:
    train_idx, test_idx = stratified_split(y, test_frac, rng)
    reg = ScoreRegressor()
    reg.fit(X[train_idx], list(y[train_idx]))

    y_true = y[test_idx]
    y_pred = np.array([reg.predict_score(X[i]) for i in test_idx])

    errors = np.abs(y_pred - y_true)
    bucket_agree = np.mean([score_to_bucket(p) == score_to_bucket(a) for p, a in zip(y_pred, y_true)])
    r = float("nan")
    if len(y_true) > 1 and np.std(y_pred) > 0:
        r, _ = pearsonr(y_pred, y_true)

    return {
        "mae": float(np.mean(errors)),
        "pearson_r": r,
        "bucket_agreement": float(bucket_agree),
        "n_test": len(test_idx),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name (substring match) or exact session id")
    parser.add_argument("--splits", type=int, default=20, help="number of repeated random splits to average over")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    session_id, name = await resolve_session(args.model)
    training_data = await get_training_data(session_id)
    if not training_data:
        raise SystemExit(f"no human-graded images for model {name!r}")

    X = np.stack([deserialize_embedding(emb) for emb, _ in training_data])
    y = np.array([score for _, score in training_data], dtype=float)
    buckets = Counter(score_to_bucket(s) for s in y)
    print(f"model: {name} ({session_id})")
    print(f"human-graded examples: {len(y)} -- score range [{y.min():.0f}, {y.max():.0f}], buckets {dict(buckets)}")
    for bucket in HUMAN_BUCKETS:
        if buckets.get(bucket, 0) < 2:
            print(f"WARNING: only {buckets.get(bucket, 0)} examples in the {bucket!r} bucket -- splits may be unstable")
    print()

    rng = np.random.default_rng(args.seed)
    results = [evaluate_once(X, y, args.test_frac, rng) for _ in range(args.splits)]

    maes = [r["mae"] for r in results]
    corrs = [r["pearson_r"] for r in results]
    agrees = [r["bucket_agreement"] for r in results]

    print(f"Over {args.splits} random splits ({args.test_frac:.0%} held out each time):")
    print(f"  MAE (0-100 scale):   mean={np.mean(maes):.2f}  std={np.std(maes):.2f}")
    print(f"  Pearson correlation: mean={np.nanmean(corrs):.3f}  std={np.nanstd(corrs):.3f}")
    print(f"  Bucket agreement:    mean={np.mean(agrees):.3f}  std={np.std(agrees):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
