from typing import Optional

from fastapi import APIRouter, HTTPException, Response

from app.labels import resolve_labels
from app.models import (
    Bucket,
    GallerySort,
    GradeResponse,
    ModelCreateRequest,
    ModelImage,
    ModelImagesPage,
    ModelRenameRequest,
    ModelSummary,
    NextAutoImageResponse,
    PromoteImageRequest,
    RegressorTestResult,
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


GALLERY_PAGE_SIZE = 25


@router.get("/{session_id}/images", response_model=ModelImagesPage)
async def get_model_images(
    session_id: str,
    bucket: Optional[Bucket] = None,
    source: Optional[str] = None,
    offset: int = 0,
    sort: GallerySort = "newest",
) -> ModelImagesPage:
    labels = resolve_labels(bucket, source)
    try:
        rows, total = await session_manager.get_images(session_id, labels, GALLERY_PAGE_SIZE, offset, sort)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    items = [
        ModelImage(
            image_id=row["id"],
            label=row["label"],
            score=row["score"],
            created_at=row["created_at"],
            graded_at=row["graded_at"],
        )
        for row in rows
    ]
    return ModelImagesPage(items=items, total=total)


@router.get("/{session_id}/test-regressor", response_model=list[RegressorTestResult])
async def test_regressor(session_id: str) -> list[RegressorTestResult]:
    try:
        results = await session_manager.test_regressor(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return [RegressorTestResult(**r) for r in results]


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
        result = await session_manager.promote_auto_image(session_id, req.image_id, req.score)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model or image")
    return GradeResponse(**result)
