from fastapi import APIRouter, HTTPException

from app.models import ModelCreateRequest, ModelImage, ModelRenameRequest, ModelSummary
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
async def get_model_images(session_id: str, label: str | None = None) -> list[ModelImage]:
    try:
        rows = await session_manager.get_images(session_id, label)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return [
        ModelImage(image_id=row["id"], label=row["label"], created_at=row["created_at"], graded_at=row["graded_at"])
        for row in rows
    ]
