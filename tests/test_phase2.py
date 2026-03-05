"""
Phase 2 tests: AudioCache + NetworkMonitor

Covers:
  - SQLite-backed audio queue: enqueue, crash recovery, completion, retries, failures
  - Background network monitor: state detection, callbacks, stop behaviour

All tests use temporary directories/files to avoid polluting the project tree.
Socket operations are patched for deterministic network state tests.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_cache import AudioCache, MAX_RETRIES  # noqa: E402
from network_monitor import NetworkMonitor       # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str) -> None:
    """Write a minimal valid WAV file to *path*."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)  # 10 ms of silence


# ---------------------------------------------------------------------------
# AudioCache tests (1–9)
# ---------------------------------------------------------------------------

class TestAudioCache(unittest.TestCase):

    def setUp(self):
        """Each test gets its own temp dir with a fresh DB and a temp WAV."""
        self._tmpdir = tempfile.mkdtemp(prefix="ac_test_")
        self._db_path = os.path.join(self._tmpdir, "test_queue.db")
        self._wav_path = os.path.join(self._tmpdir, "sample.wav")
        _make_wav(self._wav_path)
        self._cache = AudioCache(db_path=self._db_path)

    def tearDown(self):
        # Best-effort cleanup
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # 1. enqueue and get_next_pending round-trip
    def test_enqueue_and_retrieve(self):
        row_id = self._cache.enqueue(self._wav_path)
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

        entry = self._cache.get_next_pending()
        self.assertIsNotNone(entry, "get_next_pending() returned None after enqueue")
        self.assertEqual(entry["audio_path"], self._wav_path)
        # Entry moved to IN_FLIGHT — no more pending
        self.assertIsNone(self._cache.get_next_pending())

    # 2. Crash recovery: IN_FLIGHT → PENDING on new AudioCache instance
    def test_crash_recovery(self):
        row_id = self._cache.enqueue(self._wav_path)
        # Consume entry (moves to IN_FLIGHT)
        self._cache.get_next_pending()

        # Verify it is IN_FLIGHT by checking no pending entries
        self.assertIsNone(self._cache.get_next_pending())

        # Simulate crash: create fresh AudioCache pointing at same DB
        recovered = AudioCache(db_path=self._db_path)
        entry = recovered.get_next_pending()
        self.assertIsNotNone(entry, "IN_FLIGHT entry was not reset to PENDING on recovery")
        self.assertEqual(entry["id"], row_id)

    # 3. mark_complete deletes the audio file
    def test_mark_complete_deletes_file(self):
        self.assertTrue(os.path.exists(self._wav_path), "WAV must exist before test")
        row_id = self._cache.enqueue(self._wav_path)
        self._cache.get_next_pending()  # moves to IN_FLIGHT
        self._cache.mark_complete(row_id)
        self.assertFalse(
            os.path.exists(self._wav_path),
            "Audio file should be deleted after mark_complete",
        )

    # 4. DB status is COMPLETE even when file was already deleted
    def test_mark_complete_updates_db_first(self):
        row_id = self._cache.enqueue(self._wav_path)
        self._cache.get_next_pending()
        # Delete file manually before calling mark_complete
        os.remove(self._wav_path)
        # Should NOT raise even though file is gone
        self._cache.mark_complete(row_id)
        # Verify DB status via a new connection (reconnect)
        recovered = AudioCache(db_path=self._db_path)
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status FROM audio_queue WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "COMPLETE")

    # 5. retry_delay_for increases with retry count and is capped at 300
    def test_retry_backoff_increases(self):
        d0 = AudioCache.retry_delay_for(0)
        d3 = AudioCache.retry_delay_for(3)
        d_max = AudioCache.retry_delay_for(100)  # should be capped

        self.assertLess(d0, d3, "retry_delay_for(0) should be < retry_delay_for(3)")
        self.assertLessEqual(d_max, 300.0 * 1.2 + 1e-9,
                             "retry_delay_for should be capped near RETRY_MAX_DELAY")

    # 6. After MAX_RETRIES calls to mark_pending, get_next_pending returns None
    def test_max_retries_excludes_from_pending(self):
        row_id = self._cache.enqueue(self._wav_path)
        for i in range(MAX_RETRIES):
            # Need entry to be IN_FLIGHT before mark_pending
            entry = self._cache.get_next_pending()
            if entry is None:
                # Already auto-failed on a previous mark_pending call
                break
            self._cache.mark_pending(entry["id"], error=f"err{i}")

        result = self._cache.get_next_pending()
        self.assertIsNone(
            result,
            f"After {MAX_RETRIES} retries get_next_pending should return None",
        )

    # 7. mark_failed — status is FAILED and file still exists
    def test_mark_failed(self):
        row_id = self._cache.enqueue(self._wav_path)
        self._cache.get_next_pending()
        self._cache.mark_failed(row_id, error="permanent failure")

        self.assertTrue(
            os.path.exists(self._wav_path),
            "Audio file must NOT be deleted after mark_failed",
        )
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status FROM audio_queue WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "FAILED")

    # 8. pending_count reflects enqueued items
    def test_pending_count(self):
        self.assertEqual(self._cache.pending_count(), 0)
        for i in range(3):
            wav = os.path.join(self._tmpdir, f"clip_{i}.wav")
            _make_wav(wav)
            self._cache.enqueue(wav)
        self.assertEqual(self._cache.pending_count(), 3)

    # 9. enqueue with a non-existent path raises an error
    def test_enqueue_nonexistent_file_raises(self):
        with self.assertRaises((FileNotFoundError, AssertionError)):
            self._cache.enqueue("/nonexistent/path/audio.wav")


# ---------------------------------------------------------------------------
# NetworkMonitor tests (10–13)
# ---------------------------------------------------------------------------

class TestNetworkMonitor(unittest.TestCase):

    # 10. is_connected property returns a bool (not None) at startup
    def test_initial_state(self):
        monitor = NetworkMonitor()
        result = monitor.is_connected
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bool)

    # 11. on_disconnected fires when all socket probes fail
    def test_callback_on_disconnect(self):
        disconnected_event = threading.Event()

        def on_disc():
            disconnected_event.set()

        # Force monitor to start as "connected" then see failure
        monitor = NetworkMonitor(
            on_disconnected=on_disc,
            check_interval=0.05,
            probe_timeout=0.1,
            endpoints=[("127.0.0.1", 1)],  # port 1 = almost certainly closed
        )
        # Prime the monitor as "connected" so the state change fires
        monitor._connected = True

        with patch("network_monitor._probe_endpoint", return_value=False):
            monitor.start()
            fired = disconnected_event.wait(timeout=2.0)
            monitor.stop()

        self.assertTrue(fired, "on_disconnected callback was not called when probes fail")

    # 12. on_connected fires when socket succeeds after previous failure
    def test_callback_on_reconnect(self):
        connected_event = threading.Event()

        def on_conn():
            connected_event.set()

        # Start as disconnected so the transition to connected fires the callback
        monitor = NetworkMonitor(
            on_connected=on_conn,
            check_interval=0.05,
            probe_timeout=0.1,
            endpoints=[("8.8.8.8", 53)],
        )
        monitor._connected = False  # prime as disconnected

        with patch("network_monitor._probe_endpoint", return_value=True):
            monitor.start()
            fired = connected_event.wait(timeout=2.0)
            monitor.stop()

        self.assertTrue(fired, "on_connected callback was not called when probe succeeds")

    # 13. stop() causes the thread to join cleanly
    def test_stop(self):
        monitor = NetworkMonitor(
            check_interval=0.05,
            probe_timeout=0.1,
            endpoints=[("127.0.0.1", 1)],
        )
        with patch("network_monitor._probe_endpoint", return_value=False):
            monitor.start()
            self.assertIsNotNone(monitor._thread)
            self.assertTrue(monitor._thread.is_alive())
            monitor.stop()

        self.assertIsNone(monitor._thread, "Thread reference should be None after stop()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
