from typing import Literal, Optional

from pydantic import BaseModel

Mode = Literal["supervised", "unsupervised"]
Label = Literal["not_part", "good", "great"]


class CrawlStartRequest(BaseModel):
    session_id: str
    seed_urls: list[str]
    mode: Mode


class CrawlStopRequest(BaseModel):
    session_id: str


class CrawlModeRequest(BaseModel):
    session_id: str
    mode: Mode


class CrawlStatusResponse(BaseModel):
    status: str
    mode: str
    pages_visited: int
    images_found: int
    images_queued: int
    images_graded: int
    images_auto_filed: int
    current_url: Optional[str]
    class_counts: dict[str, int]


class Prediction(BaseModel):
    label: str
    probs: dict[str, float]


class NextImageResponse(BaseModel):
    image_id: str
    prediction: Optional[Prediction]


class GradeRequest(BaseModel):
    session_id: str
    image_id: str
    label: Label


class GradeResponse(BaseModel):
    status: str
    training_examples: int
    class_counts: dict[str, int]


class ModelCreateRequest(BaseModel):
    name: str


class ModelRenameRequest(BaseModel):
    name: str


class ModelSummary(BaseModel):
    session_id: str
    name: str
    created_at: str
    updated_at: str
    mode: str
    status: str
    class_counts: dict[str, int]


class ModelImage(BaseModel):
    image_id: str
    label: str
    created_at: str
    graded_at: Optional[str]
