"""Evaluate the nearest-centroid classifier against a model's own
human-graded images via repeated held-out train/test splits.

Only human-graded labels (not_part/good/great) are used -- auto-filed
(auto_not_part/auto-good/auto-great) images are the classifier's own past
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

sys.path.insert(0, ".")

from app.db import HUMAN_LABELS, get_training_data, list_sessions  # noqa: E402
from app.ml.classifier import ClassifierHead  # noqa: E402
from app.ml.embeddings import deserialize_embedding  # noqa: E402


async def resolve_session(query: str) -> tuple[str, str]:
    sessions = await list_sessions()
    matches = [s for s in sessions if query.lower() in s["name"].lower() or query == s["id"]]
    if not matches:
        raise SystemExit(f"no model matching {query!r}. Available: {[s['name'] for s in sessions]}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous match for {query!r}: {[s['name'] for s in matches]}")
    return matches[0]["id"], matches[0]["name"]


def stratified_split(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Splits indices so each class is represented proportionally in both
    train and test -- a plain random split can otherwise leave a sparse
    class entirely out of one side."""
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def evaluate_once(X: np.ndarray, y: np.ndarray, test_frac: float, rng: np.random.Generator) -> dict:
    train_idx, test_idx = stratified_split(y, test_frac, rng)
    clf = ClassifierHead()
    clf.fit(X[train_idx], list(y[train_idx]))

    y_true = y[test_idx]
    y_pred = np.array([clf.predict_label(X[i]) for i in test_idx])

    positive = np.array([label != "not_part" for label in y_true])
    pred_positive = np.array([label != "not_part" for label in y_pred])

    false_negatives = int(np.sum(positive & ~pred_positive))  # actually in-class, predicted not_part
    false_positives = int(np.sum(~positive & pred_positive))  # actually not_part, predicted in-class
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "n_test": len(test_idx),
        "false_negative_rate": false_negatives / n_positive if n_positive else float("nan"),
        "false_positive_rate": false_positives / n_negative if n_negative else float("nan"),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "confusion": Counter(zip(y_true.tolist(), y_pred.tolist())),
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
    y = np.array([label for _, label in training_data])
    counts = Counter(y.tolist())
    print(f"model: {name} ({session_id})")
    print(f"human-graded examples: {len(y)} -- {dict(counts)}")
    for label in HUMAN_LABELS:
        if counts.get(label, 0) < 2:
            print(f"WARNING: only {counts.get(label, 0)} examples of {label!r} -- splits may be unstable")
    print()

    rng = np.random.default_rng(args.seed)
    results = [evaluate_once(X, y, args.test_frac, rng) for _ in range(args.splits)]

    accs = [r["accuracy"] for r in results]
    fnrs = [r["false_negative_rate"] for r in results]
    fprs = [r["false_positive_rate"] for r in results]

    print(f"Over {args.splits} random stratified splits ({args.test_frac:.0%} held out each time):")
    print(f"  accuracy:            mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
    print(f"  false negative rate: mean={np.mean(fnrs):.3f}  std={np.std(fnrs):.3f}   (in-class predicted not_part)")
    print(f"  false positive rate: mean={np.mean(fprs):.3f}  std={np.std(fprs):.3f}   (not_part predicted in-class)")
    print()

    confusion_total: Counter = Counter()
    for r in results:
        confusion_total.update(r["confusion"])
    labels = sorted(set(y.tolist()))
    print("Aggregate confusion matrix (rows=true, cols=predicted, summed over all splits):")
    header = "true\\pred".ljust(12) + "".join(label.ljust(12) for label in labels)
    print(header)
    for true_label in labels:
        row = true_label.ljust(12)
        for pred_label in labels:
            row += str(confusion_total.get((true_label, pred_label), 0)).ljust(12)
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
