import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from app import db
from app.config import EMBEDDING_MODEL_ID, NEXT_IMAGE_LONG_POLL_TIMEOUT, PENDING_QUEUE_MAXSIZE
from app.crawler.crawler import CrawledImage, Crawler
from app.crawler.dedup import ContentHashIndex
from app.ml.classifier import ClassifierHead
from app.ml.embeddings import deserialize_embedding, serialize_embedding
from app.storage.image_store import persist_graded_image, should_persist
from app.storage.paths import model_path


def _to_auto_label(predicted: Optional[str]) -> str:
    """Maps a classifier prediction to its auto-filed label/folder name.
    not_part keeps its existing underscore form; good/great use a hyphen."""
    if predicted is None or predicted == "not_part":
        return "auto_not_part"
    return f"auto-{predicted}"


class SessionState:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.classifier = ClassifierHead()
        # Populated in _get_or_create_state, which knows this session's
        # already-seen content hashes; empty here since building it needs a
        # DB round trip.
        self.dedup_index = ContentHashIndex()
        self.pending_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=PENDING_QUEUE_MAXSIZE)
        self.pending_bytes: dict[str, bytes] = {}
        self.stop_event = asyncio.Event()
        self.crawl_task: Optional[asyncio.Task] = None


class SessionManager:
    """Owns all in-memory per-model state (pending-grade queue, classifier
    head, running crawl task). A "model" is a `sessions` DB row: a durable
    identity (id + name) that can be crawled into across multiple runs over
    time. In-memory state is created lazily on first access and, if missing,
    resumes a previously-trained classifier head from disk — this is what
    lets a model survive a server restart or simply not having been touched
    yet this process lifetime."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def _state(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown session {session_id!r}")
        return state

    async def _get_or_create_state(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is not None:
            return state
        row = await db.get_session(session_id)
        if row is None:
            raise KeyError(f"unknown session {session_id!r}")
        state = SessionState(row["mode"])
        state.classifier.load(model_path(session_id))
        existing_hashes = await db.get_content_hashes(session_id)
        state.dedup_index = ContentHashIndex(existing_hashes)
        self._sessions[session_id] = state
        return state

    async def _stop_existing_crawl(self, state: SessionState) -> None:
        task = state.crawl_task
        if task is None or task.done():
            return
        state.stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

    async def create_model(self, name: str) -> str:
        return await db.create_session(name)

    @staticmethod
    async def _row_to_summary(row: dict) -> dict:
        class_counts = await db.get_class_counts(row["id"])
        return {
            "session_id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "mode": row["mode"],
            "status": row["status"],
            "class_counts": class_counts,
        }

    async def get_model(self, session_id: str) -> dict:
        row = await db.get_session(session_id)
        if row is None:
            raise KeyError(f"unknown session {session_id!r}")
        return await self._row_to_summary(row)

    async def list_models(self) -> list[dict]:
        rows = await db.list_sessions()
        return [await self._row_to_summary(row) for row in rows]

    async def rename_model(self, session_id: str, name: str) -> None:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        await db.rename_session(session_id, name)

    async def get_images(self, session_id: str, label: Optional[str] = None) -> list[dict]:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        return await db.get_images_for_session(session_id, label)

    async def start_crawl(self, session_id: str, seed_urls: list[str], mode: str) -> None:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        state = await self._get_or_create_state(session_id)
        await self._stop_existing_crawl(state)
        state.mode = mode
        state.stop_event = asyncio.Event()
        await db.start_crawl_run(session_id, seed_urls, mode)

        async def is_duplicate(content_hash: str) -> bool:
            # In-memory bloom filter + sorted-list check (see ContentHashIndex) —
            # no DB round trip needed on the crawler's hot path.
            return state.dedup_index.contains(content_hash)

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
            state.dedup_index.add(crawled.content_hash)
            if state.mode == "supervised":
                state.pending_bytes[image_id] = crawled.image_bytes
                await state.pending_queue.put(image_id)
            else:
                predicted = state.classifier.predict_label(crawled.embedding)
                label = _to_auto_label(predicted)
                local_path = None
                if should_persist(label):
                    local_path = persist_graded_image(session_id, crawled.content_hash, crawled.image_bytes, label)
                await db.set_image_label(image_id, label, local_path)

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
        state = await self._get_or_create_state(session_id)
        session_row = await db.get_session(session_id)
        class_counts = await db.get_class_counts(session_id)
        images_graded = sum(v for k, v in class_counts.items() if k in db.HUMAN_LABELS)
        images_auto_filed = sum(class_counts.values()) - images_graded
        return {
            "status": session_row["status"],
            "mode": state.mode,
            "pages_visited": session_row["pages_visited"],
            "images_found": session_row["images_found"],
            "images_queued": state.pending_queue.qsize(),
            "images_graded": images_graded,
            "images_auto_filed": images_auto_filed,
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
        # predict_label(), NOT max(predict_proba()) -- the two can disagree
        # for the hierarchical classifier (see ClassifierHead.predict_label).
        label = state.classifier.predict_label(embedding)
        prediction = None
        if label is not None:
            prediction = {"label": label, "probs": state.classifier.predict_proba(embedding)}
        return {"image_id": image_id, "prediction": prediction}

    async def get_image_bytes(self, session_id: str, image_id: str) -> Optional[bytes]:
        """Serves both in-flight grading images (in-memory, not yet graded)
        and already-persisted ones (read from disk) through one lookup —
        the latter is what makes the gallery work."""
        state = self._sessions.get(session_id)
        if state is not None:
            pending = state.pending_bytes.get(image_id)
            if pending is not None:
                return pending

        row = await db.get_image(image_id)
        if row is None or row["session_id"] != session_id or row["local_path"] is None:
            return None
        path = Path(row["local_path"])
        return path.read_bytes() if path.exists() else None

    async def grade_image(self, session_id: str, image_id: str, label: str) -> dict:
        state = self._state(session_id)
        row = await db.get_image(image_id)
        image_bytes = state.pending_bytes.pop(image_id, None)

        local_path = None
        if should_persist(label) and image_bytes is not None:
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
