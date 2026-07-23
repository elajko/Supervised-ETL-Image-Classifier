import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

from app.config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    seed_urls TEXT NOT NULL,       -- JSON array
    mode TEXT NOT NULL,            -- 'supervised' | 'unsupervised'
    status TEXT NOT NULL,          -- 'idle' | 'crawling' | 'stopped'
    pages_visited INTEGER NOT NULL DEFAULT 0,
    images_found INTEGER NOT NULL DEFAULT 0,
    current_url TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    source_page_url TEXT,
    image_url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    label TEXT,                    -- NULL | not_part | good | great | auto_not_part | auto-good | auto-great
    local_path TEXT,
    graded_at TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_images_session_hash
    ON images(session_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_images_session_label
    ON images(session_id, label);
"""

HUMAN_LABELS = ("not_part", "good", "great")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Migration guard: any pre-existing local data/app.db predates the
        # `name` column. Add it and backfill from `id` so old rows stay usable.
        async with db.execute("PRAGMA table_info(sessions)") as cur:
            cols = [row[1] for row in await cur.fetchall()]
        if "name" not in cols:
            await db.execute("ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            await db.execute("UPDATE sessions SET name = id WHERE name = ''")
            await db.commit()


async def create_session(name: str) -> str:
    session_id = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, created_at, updated_at, name, seed_urls, mode, status, "
            "pages_visited, images_found, current_url) VALUES (?, ?, ?, ?, '[]', 'supervised', 'idle', 0, 0, NULL)",
            (session_id, now, now, name),
        )
        await db.commit()
    return session_id


async def start_crawl_run(session_id: str, seed_urls: list[str], mode: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET seed_urls = ?, mode = ?, status = 'crawling', "
            "pages_visited = 0, images_found = 0, current_url = NULL, updated_at = ? WHERE id = ?",
            (json.dumps(seed_urls), mode, _now(), session_id),
        )
        await db.commit()


async def list_sessions() -> list[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def rename_session(session_id: str, name: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
            (name, _now(), session_id),
        )
        await db.commit()


async def get_session(session_id: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def update_session_status(session_id: str, status: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), session_id),
        )
        await db.commit()


async def update_session_mode(session_id: str, mode: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET mode = ?, updated_at = ? WHERE id = ?",
            (mode, _now(), session_id),
        )
        await db.commit()


async def update_session_progress(
    session_id: str,
    pages_visited: Optional[int] = None,
    images_found: Optional[int] = None,
    current_url: Optional[str] = None,
) -> None:
    fields = []
    values: list[Any] = []
    if pages_visited is not None:
        fields.append("pages_visited = ?")
        values.append(pages_visited)
    if images_found is not None:
        fields.append("images_found = ?")
        values.append(images_found)
    if current_url is not None:
        fields.append("current_url = ?")
        values.append(current_url)
    if not fields:
        return
    fields.append("updated_at = ?")
    values.append(_now())
    values.append(session_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", values)
        await db.commit()


async def get_content_hashes(session_id: str) -> list[str]:
    """All content hashes already seen for this session — used to rebuild
    the in-memory bloom filter + sorted-list dedup index when a model's
    state is (re)loaded."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT content_hash FROM images WHERE session_id = ?",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
            return [row[0] for row in rows]


async def insert_pending_image(
    session_id: str,
    source_page_url: Optional[str],
    image_url: str,
    content_hash: str,
    embedding: bytes,
    embedding_model: str,
) -> str:
    image_id = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO images (id, session_id, source_page_url, image_url, content_hash, "
            "embedding, embedding_model, label, local_path, graded_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)",
            (image_id, session_id, source_page_url, image_url, content_hash, embedding, embedding_model, now),
        )
        await db.commit()
    return image_id


async def set_image_label(image_id: str, label: str, local_path: Optional[str]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE images SET label = ?, local_path = ?, graded_at = ? WHERE id = ?",
            (label, local_path, _now(), image_id),
        )
        await db.commit()


async def get_image(image_id: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM images WHERE id = ?", (image_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_training_data(session_id: str) -> list[tuple[bytes, str]]:
    """Returns (embedding_bytes, label) pairs for human-graded images only —
    auto-labeled images are the classifier's own past predictions and must not
    be fed back into training."""
    placeholders = ", ".join("?" for _ in HUMAN_LABELS)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            f"SELECT embedding, label FROM images WHERE session_id = ? AND label IN ({placeholders})",
            (session_id, *HUMAN_LABELS),
        ) as cur:
            rows = await cur.fetchall()
            return [(row[0], row[1]) for row in rows]


async def get_human_graded_images(session_id: str) -> list[dict[str, Any]]:
    """id/embedding/label for every human-graded image -- used to test the
    current classifier against manually-verified ground truth."""
    placeholders = ", ".join("?" for _ in HUMAN_LABELS)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT id, embedding, label FROM images WHERE session_id = ? AND label IN ({placeholders})",
            (session_id, *HUMAN_LABELS),
        ) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def get_next_unreviewed_auto_image(session_id: str, auto_labels: tuple[str, ...]) -> Optional[dict[str, Any]]:
    """The oldest not-yet-reviewed auto-filed image. Once an image is
    promoted its label changes to a human one, so it naturally drops out of
    this query -- no separate "reviewed" flag needed."""
    placeholders = ", ".join("?" for _ in auto_labels)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM images WHERE session_id = ? AND label IN ({placeholders}) "
            "AND local_path IS NOT NULL ORDER BY created_at ASC LIMIT 1",
            (session_id, *auto_labels),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_images_for_session(session_id: str, labels: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Images actually persisted to disk for this session — excludes discarded
    not_part/auto_not_part rows (which never got a local_path) without needing
    a separate exclusion list."""
    query = (
        "SELECT id, label, image_url, local_path, created_at, graded_at FROM images "
        "WHERE session_id = ? AND local_path IS NOT NULL"
    )
    params: list[Any] = [session_id]
    if labels:
        placeholders = ", ".join("?" for _ in labels)
        query += f" AND label IN ({placeholders})"
        params.extend(labels)
    query += " ORDER BY graded_at DESC"
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def get_class_counts(session_id: str) -> dict[str, int]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT label, COUNT(*) FROM images WHERE session_id = ? AND label IS NOT NULL GROUP BY label",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
            return {label: count for label, count in rows}
