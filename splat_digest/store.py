from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


class Store:
    """SQLite state: cached LLM analyses + which items were already sent.

    An item is marked as sent only after Telegram accepted the message, so a
    failed run doesn't silently lose papers; cached analyses avoid paying twice.
    """

    def __init__(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS items (
                key TEXT PRIMARY KEY,
                title TEXT,
                first_seen TEXT,
                analysis TEXT,
                sent_at TEXT
            )"""
        )
        self.db.commit()

    def sent_keys(self, keys: list[str]) -> set[str]:
        if not keys:
            return set()
        q = f"SELECT key FROM items WHERE sent_at IS NOT NULL AND key IN ({','.join('?' * len(keys))})"
        return {r[0] for r in self.db.execute(q, keys)}

    def get_analysis(self, key: str) -> dict | None:
        row = self.db.execute("SELECT analysis FROM items WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def save_analysis(self, key: str, title: str, analysis: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO items(key, title, first_seen, analysis) VALUES (?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET analysis=excluded.analysis""",
            (key, title, now, json.dumps(analysis, ensure_ascii=False)),
        )
        self.db.commit()

    def mark_sent(self, keys: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.executemany(
            """INSERT INTO items(key, first_seen, sent_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET sent_at=excluded.sent_at""",
            [(k, now, now) for k in keys],
        )
        self.db.commit()
