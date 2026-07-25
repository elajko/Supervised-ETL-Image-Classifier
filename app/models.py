from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["supervised", "unsupervised"]
GallerySort = Literal["newest", "rating"]
Score = float  # 0-100, validated with Field(ge=0, le=100) on request models


class CrawlStartRequest(BaseModel):
    session_id: str
    seed_urls: list[str]
    mode: Mode
    save_threshold: Score = Field(ge=0, le=100, default=0)


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
    last_error: Optional[str] = None
    save_threshold: float = 0


class Prediction(BaseModel):
    score: float


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
    label_counts: dict[str, int]


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


class NextAutoImageResponse(BaseModel):
    image_id: str
    score: float


class PromoteImageRequest(BaseModel):
    image_id: str
    score: Score = Field(ge=0, le=100)


class SiteStat(BaseModel):
    domain: str
    average_score: float
    image_count: int


class ScoreHistogramResponse(BaseModel):
    human: list[int]
    auto: list[int]


class SourceStatus(BaseModel):
    name: str
    domains: list[str]
    needs_client_secret: bool
    supports_interactive_auth: bool
    configured: bool
    authenticated: bool


class SourceCredentialsRequest(BaseModel):
    client_id: str
    client_secret: Optional[str] = None
