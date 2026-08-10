from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ConversationRepository:
    """SQLite storage with parameterized queries and no PII fields."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                predicted_emotion TEXT,
                confidence REAL,
                intent TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                response_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                latency_ms REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                helpful INTEGER NOT NULL CHECK(helpful IN (0, 1)),
                timestamp TEXT NOT NULL,
                FOREIGN KEY(message_id) REFERENCES messages(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp);
            """)

    def add_message(self, record: dict[str, Any]) -> int:
        columns = tuple(record)
        placeholders = ", ".join("?" for _ in columns)
        with self.connect() as conn:
            cursor = conn.execute(f"INSERT INTO messages ({', '.join(columns)}) VALUES ({placeholders})", tuple(record.values()))
            return int(cursor.lastrowid)

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id", (session_id,))]

    def add_feedback(self, message_id: int, helpful: bool) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM messages WHERE id = ?", (message_id,)).fetchone()
            if not exists:
                return False
            conn.execute("INSERT INTO feedback (message_id, helpful, timestamp) VALUES (?, ?, ?)", (message_id, int(helpful), datetime.now(timezone.utc).isoformat()))
        return True

    def delete_expired(self, retention_days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
            return cursor.rowcount
