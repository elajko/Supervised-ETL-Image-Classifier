from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["supervised", "unsupervised"]
Bucket = Literal["low", "medium", "high"]
GallerySort = Literal["newest", "rating"]
Score = float  # 0-100, validated with Field(ge=0, le=100) on request models


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
    bucket_counts: dict[str, int]


class Prediction(BaseModel):
    score: float
    bucket: str


class NextImageResponse(BaseModel):
    image_id: str
    prediction: Optional[Prediction]


class GradeRequest(BaseModel):
    session_id: str
    image_id: str
    score: Score = Field(ge=0, le=100)


class GradeResponse(BaseModel):
    status: str
    training_examples: int
    bucket_counts: dict[str, int]


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
    bucket_counts: dict[str, int]


class ModelImage(BaseModel):
    image_id: str
    label: str
    score: Optional[float]
    created_at: str
    graded_at: Optional[str]


class ModelImagesPage(BaseModel):
    items: list[ModelImage]
    total: int


class RegressorTestResult(BaseModel):
    image_id: str
    predicted: Optional[float]
    actual: float
    error: Optional[float]
    bucket_agree: bool


class NextAutoImageResponse(BaseModel):
    image_id: str
    score: float
    bucket: str


class PromoteImageRequest(BaseModel):
    image_id: str
    score: Score = Field(ge=0, le=100)
