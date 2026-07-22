import asyncio
from typing import Optional

import numpy as np

from app import db
from app.config import EMBEDDING_MODEL_ID, NEXT_IMAGE_LONG_POLL_TIMEOUT, PENDING_QUEUE_MAXSIZE
from app.crawler.crawler import CrawledImage, Crawler
from app.ml.classifier import ClassifierHead
from app.ml.embeddings import deserialize_embedding, serialize_embedding
from app.storage.image_store import DISCARDED_LABELS, persist_graded_image
from app.storage.paths import model_path


class SessionState:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.classifier = ClassifierHead()
        self.pending_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=PENDING_QUEUE_MAXSIZE)
        self.pending_bytes: dict[str, bytes] = {}
        self.stop_event = asyncio.Event()
        self.crawl_task: Optional[asyncio.Task] = None
        self.images_auto_filed = 0


class SessionManager:
    """Owns all in-memory per-session state (pending-grade queue, classifier
    head, running crawl task) and wires the crawler's sink to the current
    mode. DB rows are the durable source of truth for graded data; in-memory
    state is rebuilt fresh each time a crawl is started."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def _state(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown session {session_id!r}")
        return state

    async def start_crawl(self, seed_urls: list[str], mode: str) -> str:
        session_id = await db.create_session(seed_urls, mode)
        state = SessionState(mode)
        self._sessions[session_id] = state

        async def is_duplicate(content_hash: str) -> bool:
            return await db.image_exists_by_hash(session_id, content_hash)

        async def on_progress(**kwargs) -> None:
            await db.update_session_progress(session_id, **kwargs)

        sink = self._make_sink(session_id, state)

        crawler = Crawler(
            seed_urls,
            is_duplicate=is_duplicate,
            sink=sink,
            on_progress=on_progress,
            stop_event=state.stop_event,
        )
        state.crawl_task = asyncio.create_task(self._run_crawl(session_id, crawler))
        return session_id

    async def _run_crawl(self, session_id: str, crawler: Crawler) -> None:
        try:
            await crawler.run()
        finally:
            await db.update_session_status(session_id, "stopped")

    def _make_sink(self, session_id: str, state: SessionState):
        async def sink(crawled: CrawledImage) -> None:
            image_id = await db.insert_pending_image(
                session_id,
                crawled.source_page_url,
                crawled.image_url,
                crawled.content_hash,
                serialize_embedding(crawled.embedding),
                EMBEDDING_MODEL_ID,
            )
            if state.mode == "supervised":
                state.pending_bytes[image_id] = crawled.image_bytes
                await state.pending_queue.put(image_id)
            else:
                predicted = state.classifier.predict_label(crawled.embedding)
                label = f"auto_{predicted}" if predicted is not None else "auto_not_part"
                local_path = None
                if label not in DISCARDED_LABELS:
                    local_path = persist_graded_image(session_id, crawled.content_hash, crawled.image_bytes, label)
                await db.set_image_label(image_id, label, local_path)
                state.images_auto_filed += 1

        return sink

    async def stop_crawl(self, session_id: str) -> None:
        state = self._state(session_id)
        state.stop_event.set()
        await db.update_session_status(session_id, "stopped")

    async def set_mode(self, session_id: str, mode: str) -> None:
        state = self._state(session_id)
        state.mode = mode
        await db.update_session_mode(session_id, mode)

    async def get_status(self, session_id: str) -> dict:
        state = self._state(session_id)
        session_row = await db.get_session(session_id)
        class_counts = await db.get_class_counts(session_id)
        images_graded = sum(v for k, v in class_counts.items() if k in db.HUMAN_LABELS)
        return {
            "status": session_row["status"],
            "mode": state.mode,
            "pages_visited": session_row["pages_visited"],
            "images_found": session_row["images_found"],
            "images_queued": state.pending_queue.qsize(),
            "images_graded": images_graded,
            "images_auto_filed": state.images_auto_filed,
            "current_url": session_row["current_url"],
            "class_counts": class_counts,
        }

    async def get_next_image(self, session_id: str, timeout: float = NEXT_IMAGE_LONG_POLL_TIMEOUT) -> Optional[dict]:
        state = self._state(session_id)
        try:
            image_id = await asyncio.wait_for(state.pending_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        row = await db.get_image(image_id)
        embedding = deserialize_embedding(row["embedding"])
        probs = state.classifier.predict_proba(embedding)
        prediction = None
        if probs is not None:
            prediction = {"label": max(probs, key=probs.get), "probs": probs}
        return {"image_id": image_id, "prediction": prediction}

    def get_pending_image_bytes(self, session_id: str, image_id: str) -> Optional[bytes]:
        state = self._state(session_id)
        return state.pending_bytes.get(image_id)

    async def grade_image(self, session_id: str, image_id: str, label: str) -> dict:
        state = self._state(session_id)
        row = await db.get_image(image_id)
        image_bytes = state.pending_bytes.pop(image_id, None)

        local_path = None
        if label not in DISCARDED_LABELS and image_bytes is not None:
            local_path = persist_graded_image(session_id, row["content_hash"], image_bytes, label)
        await db.set_image_label(image_id, label, local_path)

        training_data = await db.get_training_data(session_id)
        X = np.stack([deserialize_embedding(emb) for emb, _ in training_data])
        y = [lbl for _, lbl in training_data]
        state.classifier.fit(X, y)
        state.classifier.save(model_path(session_id))

        class_counts = await db.get_class_counts(session_id)
        return {"status": "ok", "training_examples": len(training_data), "class_counts": class_counts}


session_manager = SessionManager()
