"""
SQLite-backed audio queue for resilient transcription processing.
Handles network drops by persisting audio files until successfully transcribed.
"""

import os
import random
import sqlite3
import threading
from pathlib import Path
from typing import Optional

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path TEXT NOT NULL,
    language TEXT DEFAULT 'ru',
    prompt TEXT DEFAULT '',
    status TEXT DEFAULT 'PENDING',
    retry_count INTEGER DEFAULT 0,
    last_error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


class AudioCache:
    """Thread-safe SQLite-backed queue for audio files awaiting transcription."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cache_dir = Path.home() / ".whisper-voice" / "audio_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "queue.db")

        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._reset_in_flight()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def _reset_in_flight(self) -> None:
        """On startup, reset any entries stuck in IN_FLIGHT state (crash recovery)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE audio_queue
                    SET status = 'PENDING',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'IN_FLIGHT'
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, audio_path: str, language: str = "ru", prompt: str = "") -> int:
        """
        Insert a new PENDING entry into the queue.

        Args:
            audio_path: Absolute path to the audio file (must exist).
            language: BCP-47 language code passed to the STT provider.
            prompt: Optional hint/context for the transcription model.

        Returns:
            The row ID of the newly created entry.

        Raises:
            FileNotFoundError: If audio_path does not exist on disk.
        """
        assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO audio_queue (audio_path, language, prompt, status)
                    VALUES (?, ?, ?, 'PENDING')
                    """,
                    (audio_path, language, prompt),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_next_pending(self) -> Optional[dict]:
        """
        Atomically fetch the oldest PENDING entry and mark it IN_FLIGHT.

        Returns:
            A dict with all queue fields, or None if no pending entries exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM audio_queue
                    WHERE status = 'PENDING'
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                ).fetchone()

                if row is None:
                    return None

                entry_id = row["id"]
                conn.execute(
                    """
                    UPDATE audio_queue
                    SET status = 'IN_FLIGHT',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (entry_id,),
                )
                conn.commit()
                return dict(row)
            finally:
                conn.close()

    def mark_complete(self, entry_id: int) -> None:
        """
        Mark an entry as COMPLETE and delete the underlying audio file.

        The DB record is updated first; if file deletion fails the entry is
        still considered complete (avoids re-processing already-transcribed audio).
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT audio_path FROM audio_queue WHERE id = ?", (entry_id,)
                ).fetchone()

                conn.execute(
                    """
                    UPDATE audio_queue
                    SET status = 'COMPLETE',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (entry_id,),
                )
                conn.commit()

                if row:
                    try:
                        os.unlink(row["audio_path"])
                    except OSError:
                        pass  # File already gone — acceptable
            finally:
                conn.close()

    def mark_pending(self, entry_id: int, error: str) -> None:
        """
        Return an IN_FLIGHT entry to PENDING for retry.

        Increments retry_count and stores the last error message.
        If retry_count has already reached MAX_RETRIES, permanently fails instead.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT retry_count FROM audio_queue WHERE id = ?", (entry_id,)
                ).fetchone()

                if row is None:
                    return

                new_count = row["retry_count"] + 1
                if new_count >= MAX_RETRIES:
                    # Exceeded limit — fail permanently without releasing lock twice
                    conn.execute(
                        """
                        UPDATE audio_queue
                        SET status = 'FAILED',
                            retry_count = ?,
                            last_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (new_count, error, entry_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE audio_queue
                        SET status = 'PENDING',
                            retry_count = ?,
                            last_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (new_count, error, entry_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def mark_failed(self, entry_id: int, error: str) -> None:
        """
        Permanently mark an entry as FAILED (audio file is kept for manual recovery).
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE audio_queue
                    SET status = 'FAILED',
                        last_error = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (error, entry_id),
                )
                conn.commit()
            finally:
                conn.close()

    def pending_count(self) -> int:
        """Return the number of PENDING entries in the queue."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM audio_queue WHERE status = 'PENDING'"
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """
        Delete COMPLETE entries older than max_age_hours.

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM audio_queue
                    WHERE status = 'COMPLETE'
                      AND updated_at < datetime('now', ? || ' hours')
                    """,
                    (f"-{max_age_hours}",),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def retry_delay_for(retry_count: int) -> float:
        """
        Exponential backoff with 20% jitter.

        Formula: min(RETRY_BASE_DELAY * 2^n, RETRY_MAX_DELAY) * uniform(0.8, 1.2)

        Args:
            retry_count: Number of retries already attempted (0-based).

        Returns:
            Seconds to wait before the next attempt.
        """
        base = min(RETRY_BASE_DELAY * (2 ** retry_count), RETRY_MAX_DELAY)
        jitter = random.uniform(0.8, 1.2)
        return base * jitter
