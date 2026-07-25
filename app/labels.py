from typing import Optional

# Every image's `label` in the DB is just where its score came from -- a
# person (via grading or promoting an auto-filed image) or the regressor's
# own unsupervised prediction, not yet reviewed. The old discrete
# low/medium/high (+ auto-*) bucket vocabulary was a holdover from before
# scores were continuous; the exact score is always the source of truth for
# rating, so there's nothing left for a bucket to add.
HUMAN_LABEL = "human"
AUTO_LABEL = "auto"


def resolve_source_labels(source: Optional[str]) -> Optional[list[str]]:
    """Underlying label value matching a source (supervised/unsupervised)
    filter -- used together with a score-range filter on the `score`
    column."""
    if source == "supervised":
        return [HUMAN_LABEL]
    if source == "unsupervised":
        return [AUTO_LABEL]
    return None
