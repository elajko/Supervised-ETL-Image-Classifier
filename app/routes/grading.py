import io

from fastapi import APIRouter, HTTPException, Response
from PIL import Image

from app.config import NEXT_IMAGE_LONG_POLL_TIMEOUT
from app.models import GradeRequest, GradeResponse, NextImageResponse, SkipPageRequest
from app.session.session_manager import session_manager

router = APIRouter(prefix="/api")


def _guess_content_type(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return f"image/{(img.format or 'jpeg').lower()}"
    except Exception:
        return "application/octet-stream"


@router.get("/next-image", response_model=None)
async def next_image(session_id: str, timeout: float = NEXT_IMAGE_LONG_POLL_TIMEOUT):
    try:
        result = await session_manager.get_next_image(session_id, timeout=timeout)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    if result is None:
        return Response(status_code=204)
    return NextImageResponse(**result)


@router.get("/image/{image_id}")
async def get_image(image_id: str, session_id: str) -> Response:
    image_bytes = await session_manager.get_image_bytes(session_id, image_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="image not found")
    return Response(content=image_bytes, media_type=_guess_content_type(image_bytes))


@router.post("/grade", response_model=GradeResponse)
async def grade(req: GradeRequest) -> GradeResponse:
    try:
        result = await session_manager.grade_image(req.session_id, req.image_id, req.score)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return GradeResponse(**result)


@router.post("/skip-page")
async def skip_page(req: SkipPageRequest) -> dict:
    try:
        result = await session_manager.skip_page(req.session_id, req.image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session or image")
    return result
