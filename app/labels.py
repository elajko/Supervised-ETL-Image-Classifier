from typing import Optional

from app.config import BUCKET_LABELS, SCORE_BUCKET_THRESHOLDS

SOURCES = ("supervised", "unsupervised")


def score_to_bucket(score: float) -> str:
    """Maps a continuous 0-100 score to its coarse folder/gallery bucket."""
    low, high = SCORE_BUCKET_THRESHOLDS
    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"


def to_auto_bucket(bucket: Optional[str]) -> str:
    """Maps a bucket to its auto-filed label/folder name. No prediction (a
    cold, untrained regressor) is treated as low. All three buckets use a
    uniform hyphen prefix -- unlike the old not_part/good/great scheme,
    there's no legacy inconsistency to preserve for this brand-new
    vocabulary."""
    return f"auto-{bucket or 'low'}"


def label_for(bucket: str, source: str) -> str:
    """The underlying stored label string for a given (bucket, source)
    pair, e.g. ('good' -> 'high', 'unsupervised') -> 'auto-high'."""
    return bucket if source == "supervised" else to_auto_bucket(bucket)


def base_bucket(label: str) -> str:
    """Strips the auto-filed prefix, if any, returning the base bucket
    (low/medium/high) -- e.g. 'auto-high' -> 'high'."""
    if label.startswith("auto-"):
        return label[len("auto-"):]
    return label


AUTO_BUCKETS = tuple(to_auto_bucket(b) for b in BUCKET_LABELS)


def resolve_source_labels(source: Optional[str]) -> Optional[list[str]]:
    """Underlying label strings matching a source (supervised/unsupervised)
    filter alone -- used together with a score-range filter on the `score`
    column, which replaced the old discrete bucket filter."""
    if source is None:
        return None
    return [label_for(b, source) for b in BUCKET_LABELS]
