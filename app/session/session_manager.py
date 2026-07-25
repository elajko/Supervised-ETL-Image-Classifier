import asyncio
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from app import db
from app.config import EMBEDDING_MODEL_ID, IMAGES_DIR, MODELS_DIR, NEXT_IMAGE_LONG_POLL_TIMEOUT, PENDING_QUEUE_MAXSIZE
from app.crawler.crawler import CrawledImage, Crawler
from app.crawler.dedup import ContentHashIndex
from app.labels import AUTO_LABEL, HUMAN_LABEL
from app.ml.embeddings import deserialize_embedding, serialize_embedding
from app.ml.regressor import ScoreRegressor
from app.storage.image_store import move_graded_image, persist_graded_image
from app.storage.paths import model_path


class SessionState:
    def __init__(self, mode: str, save_threshold: float = 0) -> None:
        self.mode = mode
        # Only images the current regressor predicts above this score get
        # persisted to disk; everything still gets hashed into dedup_index
        # regardless, so a below-threshold image won't be re-fetched and
        # re-scored on a future crawl of the same pages.
        self.save_threshold = save_threshold
        self.regressor = ScoreRegressor()
        # Populated in _get_or_create_state, which knows this session's
        # already-seen content hashes; empty here since building it needs a
        # DB round trip.
        self.dedup_index = ContentHashIndex()
        self.pending_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=PENDING_QUEUE_MAXSIZE)
        self.pending_bytes: dict[str, bytes] = {}
        # image_id -> source_page_url, for every image currently sitting in
        # pending_queue (or just popped off it and awaiting a grade) --
        # lets "skip this page" find and drop queued images from the same
        # page without a DB round trip per item.
        self.pending_source_pages: dict[str, str] = {}
        # Page URLs the user chose to stop pulling images from mid-crawl.
        # Purely in-memory and per-crawl-run: forgotten as soon as this
        # crawl stops, unlike the dedup index which persists across runs.
        self.skipped_pages: set[str] = set()
        self.stop_event = asyncio.Event()
        self.crawl_task: Optional[asyncio.Task] = None
        # Most recent per-seed-URL failure from a source adapter (e.g. missing
        # credentials, a rejected token) -- surfaced in crawl status so the
        # user isn't left guessing why a source produced nothing.
        self.last_error: Optional[str] = None


class SessionManager:
    """Owns all in-memory per-model state (pending-grade queue, regressor,
    running crawl task). A "model" is a `sessions` DB row: a durable
    identity (id + name) that can be crawled into across multiple runs over
    time. In-memory state is created lazily on first access and, if missing,
    resumes a previously-trained regressor from disk — this is what lets a
    model survive a server restart or simply not having been touched yet
    this process lifetime."""

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
        state = SessionState(row["mode"], row["save_threshold"])
        state.regressor.load(model_path(session_id))
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
        return {
            "session_id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "mode": row["mode"],
            "status": row["status"],
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

    async def get_images(
        self,
        session_id: str,
        labels: Optional[list[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        sort: str = "newest",
        min_score: float = 0,
        max_score: float = 100,
        domain: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        rows = await db.get_images_for_session(session_id, labels, limit, offset, sort, min_score, max_score, domain)
        total = await db.count_images_for_session(session_id, labels, min_score, max_score, domain)
        return rows, total

    async def get_site_stats(self, session_id: str) -> list[dict]:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        return await db.get_site_stats(session_id)

    async def get_score_histogram(self, session_id: str) -> dict[str, list[int]]:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        return await db.get_score_histogram(session_id)

    async def test_regressor(self, session_id: str) -> list[dict]:
        """Evaluates the current regressor against every human-graded image
        for this model -- no re-embedding needed, embeddings are already
        stored, so this is just a forward pass per image."""
        state = await self._get_or_create_state(session_id)
        rows = await db.get_human_graded_images(session_id)
        results = []
        for row in rows:
            embedding = deserialize_embedding(row["embedding"])
            predicted = state.regressor.predict_score(embedding)
            actual = row["score"]
            results.append(
                {
                    "image_id": row["id"],
                    "predicted": predicted,
                    "actual": actual,
                    "error": abs(predicted - actual) if predicted is not None else None,
                }
            )
        return results

    async def start_crawl(self, session_id: str, seed_urls: list[str], mode: str, save_threshold: float = 0) -> None:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        state = await self._get_or_create_state(session_id)
        await self._stop_existing_crawl(state)
        state.mode = mode
        state.save_threshold = save_threshold
        state.skipped_pages = set()
        state.pending_source_pages = {}
        state.stop_event = asyncio.Event()
        state.last_error = None
        await db.start_crawl_run(session_id, seed_urls, mode, save_threshold)

        async def is_duplicate(content_hash: str) -> bool:
            # In-memory bloom filter + sorted-list check (see ContentHashIndex) —
            # no DB round trip needed on the crawler's hot path.
            return state.dedup_index.contains(content_hash)

        async def on_progress(**kwargs) -> None:
            await db.update_session_progress(session_id, **kwargs)

        async def on_error(url: str, message: str) -> None:
            state.last_error = f"{url}: {message}"

        sink = self._make_sink(session_id, state)

        crawler = Crawler(
            seed_urls,
            is_duplicate=is_duplicate,
            sink=sink,
            on_progress=on_progress,
            on_error=on_error,
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
            if crawled.source_page_url in state.skipped_pages:
                # The user rejected this page mid-crawl -- treat it as if
                # the crawler never found this image at all (no DB row, no
                # dedup entry), not just a low-scoring one.
                return

            image_id = await db.insert_pending_image(
                session_id,
                crawled.source_page_url,
                crawled.image_url,
                crawled.content_hash,
                serialize_embedding(crawled.embedding),
                EMBEDDING_MODEL_ID,
            )
            # Always hashed into the dedup index and given a DB row (which is
            # what the index gets rebuilt from on restart) regardless of the
            # save-threshold decision below -- a filtered-out image still
            # shouldn't be re-fetched and re-scored on a future crawl.
            state.dedup_index.add(crawled.content_hash)

            predicted = state.regressor.predict_score(crawled.embedding)
            # No prediction means an untrained regressor -- always save in
            # that case, since filtering everything out before any grading
            # exists would leave nothing to bootstrap training with.
            if predicted is not None and predicted < state.save_threshold:
                return

            if state.mode == "supervised":
                state.pending_bytes[image_id] = crawled.image_bytes
                state.pending_source_pages[image_id] = crawled.source_page_url or ""
                await state.pending_queue.put(image_id)
            else:
                score = predicted if predicted is not None else 0.0
                local_path = persist_graded_image(session_id, crawled.content_hash, crawled.image_bytes, AUTO_LABEL)
                await db.set_image_grade(image_id, AUTO_LABEL, score, local_path)

        return sink

    async def stop_crawl(self, session_id: str) -> None:
        state = self._state(session_id)
        state.stop_event.set()
        state.skipped_pages = set()
        await db.update_session_status(session_id, "stopped")

    async def skip_page(self, session_id: str, image_id: str) -> dict:
        """Rejects the page the given (currently-displayed) image came
        from: drops that image plus every other already-queued image from
        the same page, and stops any more images from that page being
        queued for the rest of this crawl. Links found on the page are
        still followed as normal -- only image extraction from it stops."""
        state = self._state(session_id)
        row = await db.get_image(image_id)
        if row is None or row["session_id"] != session_id:
            raise KeyError(f"unknown image {image_id!r}")

        page_url = row["source_page_url"]
        if not page_url:
            return {"status": "ok", "removed": 0}
        state.skipped_pages.add(page_url)

        state.pending_bytes.pop(image_id, None)
        state.pending_source_pages.pop(image_id, None)
        await db.delete_image(image_id)
        removed = 1

        # Drain synchronously (no awaits) so a concurrent sink() call can't
        # race a put() into the middle of this rebuild; batch the DB
        # deletes for the dropped ones afterward.
        drained = []
        while True:
            try:
                drained.append(state.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        to_delete = []
        for queued_id in drained:
            if state.pending_source_pages.get(queued_id) == page_url:
                state.pending_bytes.pop(queued_id, None)
                state.pending_source_pages.pop(queued_id, None)
                to_delete.append(queued_id)
            else:
                state.pending_queue.put_nowait(queued_id)

        for queued_id in to_delete:
            await db.delete_image(queued_id)
        removed += len(to_delete)

        return {"status": "ok", "removed": removed}

    async def set_mode(self, session_id: str, mode: str) -> None:
        state = self._state(session_id)
        state.mode = mode
        await db.update_session_mode(session_id, mode)

    async def get_status(self, session_id: str) -> dict:
        state = await self._get_or_create_state(session_id)
        session_row = await db.get_session(session_id)
        label_counts = await db.get_label_counts(session_id)
        return {
            "status": session_row["status"],
            "mode": state.mode,
            "pages_visited": session_row["pages_visited"],
            "images_found": session_row["images_found"],
            "images_queued": state.pending_queue.qsize(),
            "images_graded": label_counts.get(HUMAN_LABEL, 0),
            "images_auto_filed": label_counts.get(AUTO_LABEL, 0),
            "current_url": session_row["current_url"],
            "last_error": state.last_error,
            "save_threshold": session_row["save_threshold"],
        }

    async def get_next_image(self, session_id: str, timeout: float = NEXT_IMAGE_LONG_POLL_TIMEOUT) -> Optional[dict]:
        state = self._state(session_id)
        try:
            image_id = await asyncio.wait_for(state.pending_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        row = await db.get_image(image_id)
        embedding = deserialize_embedding(row["embedding"])
        predicted = state.regressor.predict_score(embedding)
        prediction = None
        if predicted is not None:
            prediction = {"score": predicted}
        return {"image_id": image_id, "prediction": prediction, "source_page_url": row["source_page_url"]}

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

    async def grade_image(self, session_id: str, image_id: str, score: float) -> dict:
        state = self._state(session_id)
        row = await db.get_image(image_id)
        image_bytes = state.pending_bytes.pop(image_id, None)

        local_path = None
        if image_bytes is not None:
            local_path = persist_graded_image(session_id, row["content_hash"], image_bytes, HUMAN_LABEL)
        await db.set_image_grade(image_id, HUMAN_LABEL, score, local_path)

        return await self._retrain_and_summarize(session_id, state)

    async def get_next_auto_image(self, session_id: str) -> Optional[dict]:
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        row = await db.get_next_unreviewed_auto_image(session_id)
        if row is None:
            return None
        return {"image_id": row["id"], "score": row["score"]}

    async def promote_auto_image(self, session_id: str, image_id: str, score: float) -> dict:
        """Assigns a new score to any already-persisted image (auto-filed or
        already human-graded), moving it into the human-graded folder so it
        always counts as supervised ground truth. Used both for confirming/
        correcting auto-filed images during review and for manually
        reclassifying an existing gallery image."""
        state = await self._get_or_create_state(session_id)
        row = await db.get_image(image_id)
        if row is None or row["session_id"] != session_id:
            raise KeyError(f"unknown image {image_id!r}")

        old_path = Path(row["local_path"]) if row["local_path"] else None
        new_path = move_graded_image(session_id, row["content_hash"], old_path, HUMAN_LABEL)
        await db.set_image_grade(image_id, HUMAN_LABEL, score, new_path)

        return await self._retrain_and_summarize(session_id, state)

    async def delete_image(self, session_id: str, image_id: str) -> dict:
        """Removes an image's file (if any) and DB row entirely. Retrains,
        since a deleted image may have been part of the training set. Note
        this doesn't retroactively un-hash the image from the in-memory
        dedup index (bloom filters can't remove entries) -- it'll be dropped
        from the index on the next server restart, once the DB row it's
        rebuilt from is gone."""
        state = await self._get_or_create_state(session_id)
        row = await db.get_image(image_id)
        if row is None or row["session_id"] != session_id:
            raise KeyError(f"unknown image {image_id!r}")

        if row["local_path"]:
            path = Path(row["local_path"])
            if path.exists():
                path.unlink()
        state.pending_bytes.pop(image_id, None)
        await db.delete_image(image_id)

        return await self._retrain_and_summarize(session_id, state)

    async def delete_model(self, session_id: str) -> None:
        """Deletes a model entirely: every image file and the regressor on
        disk, plus the session and image rows in the DB. Stops any active
        crawl first so nothing is still writing into the directory being
        removed."""
        if await db.get_session(session_id) is None:
            raise KeyError(f"unknown session {session_id!r}")
        state = self._sessions.pop(session_id, None)
        if state is not None:
            await self._stop_existing_crawl(state)

        images_dir = IMAGES_DIR / session_id
        if images_dir.exists():
            shutil.rmtree(images_dir)
        model_dir = MODELS_DIR / session_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

        await db.delete_session(session_id)

    async def _retrain_and_summarize(self, session_id: str, state: SessionState) -> dict:
        training_data = await db.get_training_data(session_id)
        X = np.stack([deserialize_embedding(emb) for emb, _ in training_data])
        y = [score for _, score in training_data]
        state.regressor.fit(X, y)
        state.regressor.save(model_path(session_id))

        label_counts = await db.get_label_counts(session_id)
        return {"status": "ok", "training_examples": len(training_data), "label_counts": label_counts}


session_manager = SessionManager()
