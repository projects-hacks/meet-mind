"""SQLite storage for meeting events, state persistence, and history."""

import json
import sqlite3
import logging
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from typing import Any

from backend.core.config import MeetingState

logger = logging.getLogger(__name__)

DB_PATH = Path("data/meetmind.db")


class MeetingDB:
    """Local SQLite storage — single file, no server, survives restarts.

    Single Responsibility: persistence only.
    """

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS meeting_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
            """)

    def log_event(self, timestamp: str, event_type: str, data: dict[str, Any]):
        """Log any event (perception, action, decision, gap)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
                (timestamp, event_type, json.dumps(data)),
            )

    def save_state(self, state: MeetingState):
        """Persist current meeting state (upsert)."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO meeting_state (id, state, updated_at)
                   VALUES (1, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(id) DO UPDATE SET state=excluded.state, updated_at=CURRENT_TIMESTAMP""",
                (json.dumps(asdict(state)),),
            )

    def load_state(self) -> MeetingState | None:
        """Load persisted meeting state, if any."""
        with self._connect() as conn:
            row = conn.execute("SELECT state FROM meeting_state WHERE id=1").fetchone()
            if row:
                data = json.loads(row["state"])
                return MeetingState(**data)
        return None

    def get_recent_events(
        self,
        event_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent events, optionally filtered by type."""
        with self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM events WHERE event_type=? ORDER BY id DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def clear(self):
        """Reset database for a new meeting session."""
        with self._connect() as conn:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM meeting_state")
        logger.info("Database cleared for new session.")
