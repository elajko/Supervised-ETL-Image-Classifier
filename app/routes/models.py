from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Response

from app.labels import resolve_labels
from app.models import (
    ClassifierTestResult,
    GradeResponse,
    ModelCreateRequest,
    ModelImage,
    ModelRenameRequest,
    ModelSummary,
    NextAutoImageResponse,
    PromoteImageRequest,
)
from app.session.session_manager import session_manager

router = APIRouter(prefix="/api/models")


@router.post("", response_model=ModelSummary)
async def create_model(req: ModelCreateRequest) -> ModelSummary:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be blank")
    session_id = await session_manager.create_model(name)
    return await session_manager.get_model(session_id)


@router.get("", response_model=list[ModelSummary])
async def list_models() -> list[ModelSummary]:
    return await session_manager.list_models()


@router.patch("/{session_id}")
async def rename_model(session_id: str, req: ModelRenameRequest) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be blank")
    try:
        await session_manager.rename_model(session_id, name)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return {"status": "ok"}


@router.get("/{session_id}/images", response_model=list[ModelImage])
async def get_model_images(
    session_id: str,
    classification: Optional[Literal["not_part", "good", "great"]] = None,
    source: Optional[Literal["supervised", "unsupervised"]] = None,
) -> list[ModelImage]:
    labels = resolve_labels(classification, source)
    try:
        rows = await session_manager.get_images(session_id, labels)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return [
        ModelImage(image_id=row["id"], label=row["label"], created_at=row["created_at"], graded_at=row["graded_at"])
        for row in rows
    ]


@router.get("/{session_id}/test-classifier", response_model=list[ClassifierTestResult])
async def test_classifier(session_id: str) -> list[ClassifierTestResult]:
    try:
        results = await session_manager.test_classifier(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return [ClassifierTestResult(**r) for r in results]


@router.get("/{session_id}/next-auto-image", response_model=None)
async def next_auto_image(session_id: str):
    try:
        result = await session_manager.get_next_auto_image(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    if result is None:
        return Response(status_code=204)
    return NextAutoImageResponse(**result)


@router.post("/{session_id}/promote-image", response_model=GradeResponse)
async def promote_image(session_id: str, req: PromoteImageRequest) -> GradeResponse:
    try:
        result = await session_manager.promote_auto_image(session_id, req.image_id, req.label)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model or image")
    return GradeResponse(**result)
