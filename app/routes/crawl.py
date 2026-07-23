from fastapi import APIRouter, HTTPException

from app.models import (
    CrawlModeRequest,
    CrawlStartRequest,
    CrawlStatusResponse,
    CrawlStopRequest,
)
from app.session.session_manager import session_manager

router = APIRouter(prefix="/api/crawl")


@router.post("/start")
async def start_crawl(req: CrawlStartRequest) -> dict:
    try:
        await session_manager.start_crawl(req.session_id, req.seed_urls, req.mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown model")
    return {"status": "crawling"}


@router.post("/stop")
async def stop_crawl(req: CrawlStopRequest) -> dict:
    try:
        await session_manager.stop_crawl(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"status": "stopped"}


@router.post("/mode")
async def set_mode(req: CrawlModeRequest) -> dict:
    try:
        await session_manager.set_mode(req.session_id, req.mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"status": "ok", "mode": req.mode}


@router.get("/status", response_model=CrawlStatusResponse)
async def crawl_status(session_id: str) -> CrawlStatusResponse:
    try:
        status = await session_manager.get_status(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return CrawlStatusResponse(**status)
