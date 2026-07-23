from typing import Optional

BASE_LABELS = ("not_part", "good", "great")
SOURCES = ("supervised", "unsupervised")


def to_auto_label(base_label: Optional[str]) -> str:
    """Maps a classifier prediction to its auto-filed label/folder name. No
    prediction (a cold, untrained classifier) is treated as not_part. not_part
    keeps its existing underscore form; good/great use a hyphen (historical
    naming, kept for compatibility with already-persisted data)."""
    if base_label is None or base_label == "not_part":
        return "auto_not_part"
    return f"auto-{base_label}"


def label_for(classification: str, source: str) -> str:
    """The underlying stored label string for a given (classification,
    source) pair, e.g. ('good', 'unsupervised') -> 'auto-good'."""
    return classification if source == "supervised" else to_auto_label(classification)


def base_classification(label: str) -> str:
    """Strips the auto-filed prefix, if any, returning the base
    classification (not_part/good/great) -- e.g. 'auto-good' -> 'good'."""
    if label == "auto_not_part":
        return "not_part"
    if label.startswith("auto-"):
        return label[len("auto-"):]
    return label


AUTO_LABELS = tuple(to_auto_label(b) for b in BASE_LABELS)


def resolve_labels(classification: Optional[str], source: Optional[str]) -> Optional[list[str]]:
    """Underlying label strings matching a gallery filter. Either axis may be
    None to mean "all" for that axis. Returns None (no filtering) only when
    both axes are unfiltered."""
    if classification is None and source is None:
        return None
    classifications = [classification] if classification else BASE_LABELS
    sources = [source] if source else SOURCES
    return [label_for(c, s) for c in classifications for s in sources]
