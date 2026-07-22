import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT
from app.db import init_db
from app.ml.embeddings import embed_image
from app.routes.crawl import router as crawl_router
from app.routes.grading import router as grading_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # warm up the CLIP model singleton so the first real request isn't slow
    await asyncio.to_thread(_warmup_clip)
    yield


def _warmup_clip() -> None:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    embed_image(buf.getvalue())


app = FastAPI(lifespan=lifespan)
app.include_router(crawl_router)
app.include_router(grading_router)
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
