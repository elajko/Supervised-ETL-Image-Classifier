"""One-time migration: remap the old discrete not_part/good/great (+ auto-*
variants) labels to the new continuous 0-100 score + low/medium/high bucket
vocabulary, moving files on disk to match.

Mapping: not_part/auto_not_part -> score 0 (bucket low), good/auto-good ->
score 60 (bucket medium), great/auto-great -> score 95 (bucket high). These
land cleanly inside the low/medium/high thresholds in app/config.py.

DRY RUN BY DEFAULT. Prints a full before/after summary -- old label counts,
new bucket counts, any local_path that doesn't exist on disk -- without
writing anything. Pass --apply to actually update the DB and move files.
Verifies row count and file existence again after applying.

Usage:
    .venv/bin/python3 scripts/migrate_to_scores.py <model name or id substring> [--apply]
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

import aiosqlite

sys.path.insert(0, ".")

from app.config import DB_PATH  # noqa: E402
from app.db import list_sessions  # noqa: E402
from app.storage.image_store import move_graded_image  # noqa: E402

OLD_TO_NEW = {
    "not_part": ("low", 0.0),
    "good": ("medium", 60.0),
    "great": ("high", 95.0),
    "auto_not_part": ("auto-low", 0.0),
    "auto-good": ("auto-medium", 60.0),
    "auto-great": ("auto-high", 95.0),
}


async def resolve_session(query: str) -> tuple[str, str]:
    sessions = await list_sessions()
    matches = [s for s in sessions if query.lower() in s["name"].lower() or query == s["id"]]
    if not matches:
        raise SystemExit(f"no model matching {query!r}. Available: {[s['name'] for s in sessions]}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous match for {query!r}: {[s['name'] for s in matches]}")
    return matches[0]["id"], matches[0]["name"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name (substring match) or exact session id")
    parser.add_argument("--apply", action="store_true", help="actually write changes (default: dry run)")
    args = parser.parse_args()

    session_id, name = await resolve_session(args.model)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ", ".join("?" for _ in OLD_TO_NEW)
        async with db.execute(
            f"SELECT * FROM images WHERE session_id = ? AND label IN ({placeholders})",
            (session_id, *OLD_TO_NEW.keys()),
        ) as cur:
            rows = await cur.fetchall()

        print(f"model: {name} ({session_id})")
        print(f"rows to migrate: {len(rows)}")
        print(f"old label counts: {dict(Counter(r['label'] for r in rows))}")

        missing_files = [r for r in rows if r["local_path"] and not Path(r["local_path"]).exists()]
        if missing_files:
            print(f"WARNING: {len(missing_files)} rows have a local_path that doesn't exist on disk:")
            for r in missing_files[:10]:
                print(f"  {r['id']}: {r['local_path']}")

        planned_new_labels = Counter(OLD_TO_NEW[r["label"]][0] for r in rows)
        print(f"planned new bucket counts: {dict(planned_new_labels)}")

        if not args.apply:
            print("\nDRY RUN -- no changes written. Re-run with --apply to execute.")
            return

        print("\nApplying...")
        moved = 0
        for r in rows:
            new_label, new_score = OLD_TO_NEW[r["label"]]
            old_path = Path(r["local_path"]) if r["local_path"] else None
            new_local_path = None
            if old_path:
                new_local_path = move_graded_image(session_id, r["content_hash"], old_path, new_label)
                moved += 1
            await db.execute(
                "UPDATE images SET label = ?, score = ?, local_path = ? WHERE id = ?",
                (new_label, new_score, new_local_path, r["id"]),
            )
        await db.commit()
        print(f"updated {len(rows)} rows, moved {moved} files")

    # Verify
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT label, COUNT(*) as n FROM images WHERE session_id = ? GROUP BY label", (session_id,)
        ) as cur:
            after = await cur.fetchall()
        async with db.execute(
            "SELECT local_path FROM images WHERE session_id = ? AND local_path IS NOT NULL", (session_id,)
        ) as cur:
            paths = await cur.fetchall()

    print(f"\nafter: {[(r['label'], r['n']) for r in after]}")
    still_missing = [p["local_path"] for p in paths if not Path(p["local_path"]).exists()]
    if still_missing:
        print(f"ERROR: {len(still_missing)} local_paths missing after migration: {still_missing[:10]}")
    else:
        print(f"verified: all {len(paths)} local_path files exist on disk")


if __name__ == "__main__":
    asyncio.run(main())
