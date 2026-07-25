import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import aiosqlite

from app.config import DATA_DIR, DB_PATH, SCORE_HISTOGRAM_BINS
from app.labels import AUTO_LABEL, HUMAN_LABEL

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
    label TEXT,                    -- NULL | human | auto
    score REAL,                    -- 0-100 continuous score: human-given if graded, model-predicted if auto-filed pending review
    local_path TEXT,
    domain TEXT,                   -- website (not page) this image was found on, e.g. 'example.com'
    graded_at TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_images_session_hash
    ON images(session_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_images_session_label
    ON images(session_id, label);

-- App-level credentials (client id/secret) for a source adapter, registered
-- by the developer with the site -- distinct from oauth_tokens below, which
-- holds the per-user access token obtained via the interactive login flow.
CREATE TABLE IF NOT EXISTS source_credentials (
    site TEXT PRIMARY KEY,
    client_id TEXT,
    client_secret TEXT
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    site TEXT PRIMARY KEY,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TEXT,
    obtained_at TEXT NOT NULL
);
"""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain_from_url(url: Optional[str]) -> Optional[str]:
    """The website (not page) a URL belongs to, e.g. 'example.com' -- 'www.'
    is stripped so 'www.example.com' and 'example.com' count as the same
    site."""
    if not url:
        return None
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


async def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Migration guard: any pre-existing local data/app.db predates the
        # `name` column. Add it and backfill from `id` so old rows stay usable.
        async with db.execute("PRAGMA table_info(sessions)") as cur:
            session_cols = [row[1] for row in await cur.fetchall()]
        if "name" not in session_cols:
            await db.execute("ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            await db.execute("UPDATE sessions SET name = id WHERE name = ''")
            await db.commit()
        # Migration guard: any pre-existing local data/app.db predates the
        # `score` column. Just adds the (nullable) column -- remapping
        # existing not_part/good/great labels to scores + new bucket names
        # and moving files on disk is a deliberate, reviewable, dry-run-first
        # manual step (see scripts/migrate_to_scores.py), not done blindly here.
        async with db.execute("PRAGMA table_info(images)") as cur:
            image_cols = [row[1] for row in await cur.fetchall()]
        if "score" not in image_cols:
            await db.execute("ALTER TABLE images ADD COLUMN score REAL")
            await db.commit()
        # Migration guard: any pre-existing local data/app.db predates the
        # `save_threshold` column. Defaults to 0 (SCORE_MIN), which saves
        # everything -- matches the previous, always-save behavior.
        async with db.execute("PRAGMA table_info(sessions)") as cur:
            session_cols = [row[1] for row in await cur.fetchall()]
        if "save_threshold" not in session_cols:
            await db.execute("ALTER TABLE sessions ADD COLUMN save_threshold REAL NOT NULL DEFAULT 0")
            await db.commit()
        # Migration guard: any pre-existing local data/app.db predates the
        # `domain` column. Add it and backfill from each row's
        # source_page_url so existing images stay filterable by site.
        async with db.execute("PRAGMA table_info(images)") as cur:
            image_cols = [row[1] for row in await cur.fetchall()]
        if "domain" not in image_cols:
            await db.execute("ALTER TABLE images ADD COLUMN domain TEXT")
            await db.commit()
            async with db.execute("SELECT id, source_page_url FROM images WHERE source_page_url IS NOT NULL") as cur:
                rows = await cur.fetchall()
            updates = [(_domain_from_url(url), image_id) for image_id, url in rows]
            await db.executemany("UPDATE images SET domain = ? WHERE id = ?", updates)
            await db.commit()
        # Created here (not in SCHEMA) since the `domain` column itself is
        # only guaranteed to exist once the migration guard above has run --
        # for a pre-existing DB, an index on it inside the initial
        # executescript would fail before the ALTER TABLE ever runs.
        await db.execute("CREATE INDEX IF NOT EXISTS idx_images_session_domain ON images(session_id, domain)")
        await db.commit()
        # Migration guard: `label` used to encode a 6-way discrete bucket
        # (low/medium/high, auto-low/auto-medium/auto-high) alongside the
        # continuous score -- now that score is the only source of truth
        # for rating, label just tracks where a score came from: 'human' or
        # 'auto'. Old values remap deterministically by their 'auto-'
        # prefix. Files are left where they are on disk -- local_path is
        # already the authoritative reference for every read, nothing
        # recomputes a path from label, so there's nothing to move.
        async with db.execute(
            "SELECT DISTINCT label FROM images WHERE label IS NOT NULL AND label NOT IN (?, ?)",
            (HUMAN_LABEL, AUTO_LABEL),
        ) as cur:
            legacy_labels = [row[0] for row in await cur.fetchall()]
        for old_label in legacy_labels:
            new_label = AUTO_LABEL if old_label.startswith("auto") else HUMAN_LABEL
            await db.execute("UPDATE images SET label = ? WHERE label = ?", (new_label, old_label))
        if legacy_labels:
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


async def start_crawl_run(session_id: str, seed_urls: list[str], mode: str, save_threshold: float = 0) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET seed_urls = ?, mode = ?, status = 'crawling', save_threshold = ?, "
            "pages_visited = 0, images_found = 0, current_url = NULL, updated_at = ? WHERE id = ?",
            (json.dumps(seed_urls), mode, save_threshold, _now(), session_id),
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


async def delete_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM images WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
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
    domain = _domain_from_url(source_page_url)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO images (id, session_id, source_page_url, image_url, content_hash, "
            "embedding, embedding_model, label, score, local_path, domain, graded_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, ?)",
            (image_id, session_id, source_page_url, image_url, content_hash, embedding, embedding_model, domain, now),
        )
        await db.commit()
    return image_id


async def set_image_grade(image_id: str, label: str, score: float, local_path: Optional[str]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE images SET label = ?, score = ?, local_path = ?, graded_at = ? WHERE id = ?",
            (label, score, local_path, _now(), image_id),
        )
        await db.commit()


async def get_image(image_id: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM images WHERE id = ?", (image_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def delete_image(image_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM images WHERE id = ?", (image_id,))
        await db.commit()


async def get_training_data(session_id: str) -> list[tuple[bytes, float]]:
    """Returns (embedding_bytes, score) pairs for human-graded images only —
    auto-filed images are the model's own past predictions and must not be
    fed back into training."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT embedding, score FROM images WHERE session_id = ? AND label = ?",
            (session_id, HUMAN_LABEL),
        ) as cur:
            rows = await cur.fetchall()
            return [(row[0], row[1]) for row in rows]


async def get_human_graded_images(session_id: str) -> list[dict[str, Any]]:
    """id/embedding/score for every human-graded image -- used to test the
    current regressor against manually-verified ground truth."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, embedding, score FROM images WHERE session_id = ? AND label = ?",
            (session_id, HUMAN_LABEL),
        ) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def get_next_unreviewed_auto_image(session_id: str) -> Optional[dict[str, Any]]:
    """The oldest not-yet-reviewed auto-filed image. Once an image is
    promoted its label changes to 'human', so it naturally drops out of
    this query -- no separate "reviewed" flag needed."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM images WHERE session_id = ? AND label = ? "
            "AND local_path IS NOT NULL ORDER BY created_at ASC LIMIT 1",
            (session_id, AUTO_LABEL),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_images_for_session(
    session_id: str,
    labels: Optional[list[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    sort: str = "newest",
    min_score: float = 0,
    max_score: float = 100,
    domain: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Images actually persisted to disk for this session."""
    query = (
        "SELECT id, label, score, image_url, local_path, domain, created_at, graded_at FROM images "
        "WHERE session_id = ? AND local_path IS NOT NULL AND score BETWEEN ? AND ?"
    )
    params: list[Any] = [session_id, min_score, max_score]
    if labels:
        placeholders = ", ".join("?" for _ in labels)
        query += f" AND label IN ({placeholders})"
        params.extend(labels)
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    order_clause = "score DESC, graded_at DESC" if sort == "rating" else "graded_at DESC"
    query += f" ORDER BY {order_clause}"
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def count_images_for_session(
    session_id: str,
    labels: Optional[list[str]] = None,
    min_score: float = 0,
    max_score: float = 100,
    domain: Optional[str] = None,
) -> int:
    query = "SELECT COUNT(*) FROM images WHERE session_id = ? AND local_path IS NOT NULL AND score BETWEEN ? AND ?"
    params: list[Any] = [session_id, min_score, max_score]
    if labels:
        placeholders = ", ".join("?" for _ in labels)
        query += f" AND label IN ({placeholders})"
        params.extend(labels)
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(query, params) as cur:
            (count,) = await cur.fetchone()
            return count


async def get_label_counts(session_id: str) -> dict[str, int]:
    """Count of human-graded vs. auto-filed (pending review) images."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT label, COUNT(*) FROM images WHERE session_id = ? AND label IS NOT NULL GROUP BY label",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
            return {label: count for label, count in rows}


async def get_score_histogram(session_id: str) -> dict[str, list[int]]:
    """Counts images (per label: human/auto) into SCORE_HISTOGRAM_BINS
    equal-width bins spanning 0-100, for the gallery's score-distribution
    bar. Always returns a full-length list per label (zero-filled), so the
    frontend doesn't need to know which bins are empty."""
    bin_width = 100 / SCORE_HISTOGRAM_BINS
    result = {HUMAN_LABEL: [0] * SCORE_HISTOGRAM_BINS, AUTO_LABEL: [0] * SCORE_HISTOGRAM_BINS}
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT label, MIN(?, CAST(score / ? AS INTEGER)) AS bin, COUNT(*) FROM images "
            "WHERE session_id = ? AND score IS NOT NULL AND label IN (?, ?) "
            "GROUP BY label, bin",
            (SCORE_HISTOGRAM_BINS - 1, bin_width, session_id, HUMAN_LABEL, AUTO_LABEL),
        ) as cur:
            rows = await cur.fetchall()
    for label, bin_index, count in rows:
        result[label][bin_index] = count
    return result


async def get_site_stats(session_id: str) -> list[dict[str, Any]]:
    """Average score and image count per source website (not page) for this
    session -- purely informational for now, doesn't feed back into
    crawling behavior. Computed on the fly from `images` rather than kept
    as a running aggregate, so it can never drift out of sync with grades,
    promotions, or deletions."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT domain, AVG(score), COUNT(*) FROM images "
            "WHERE session_id = ? AND score IS NOT NULL AND domain IS NOT NULL "
            "GROUP BY domain ORDER BY AVG(score) DESC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [{"domain": domain, "average_score": avg_score, "image_count": count} for domain, avg_score, count in rows]


async def get_source_credentials(site: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM source_credentials WHERE site = ?", (site,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def set_source_credentials(site: str, client_id: str, client_secret: Optional[str]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO source_credentials (site, client_id, client_secret) VALUES (?, ?, ?) "
            "ON CONFLICT(site) DO UPDATE SET client_id = excluded.client_id, client_secret = excluded.client_secret",
            (site, client_id, client_secret),
        )
        await db.commit()


async def get_oauth_token(site: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM oauth_tokens WHERE site = ?", (site,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def set_oauth_token(
    site: str, access_token: str, refresh_token: Optional[str], expires_at: Optional[str]
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO oauth_tokens (site, access_token, refresh_token, expires_at, obtained_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(site) DO UPDATE SET access_token = excluded.access_token, "
            "refresh_token = excluded.refresh_token, expires_at = excluded.expires_at, "
            "obtained_at = excluded.obtained_at",
            (site, access_token, refresh_token, expires_at, _now()),
        )
        await db.commit()
