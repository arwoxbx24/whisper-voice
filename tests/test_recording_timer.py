"""
Tests for auto-stop recording timer (max_recording_seconds feature).

Tests:
  - Timer is created when recording starts (max_sec > 0)
  - Timer is NOT created when max_recording_seconds = 0
  - Timer is cancelled on manual stop
  - Timer is cancelled on cancel recording
  - _auto_stop_recording calls _stop_recording_async
  - Countdown display via update_countdown
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(max_recording_seconds=300):
    """Create a WhisperVoiceApp with mocked dependencies."""
    from src.app import WhisperVoiceApp

    config = {
        "api_key": "",
        "language": "ru",
        "hotkey": "<ctrl>+<shift>+space",
        "hotkey_mode": "toggle",
        "mouse_button": None,
        "stt_providers": ["openai"],
        "deepgram_api_key": "",
        "local_whisper_model": "base",
        "local_whisper_device": "cpu",
        "model": "whisper-1",
        "prompt_context": "",
        "insert_method": "auto",
        "audio_cache_enabled": False,
        "sound_feedback": False,
        "auto_start": False,
        "log_level": "INFO",
        "max_recording_seconds": max_recording_seconds,
    }
    app = WhisperVoiceApp(config)

    # Mock sub-components so we don't need real hardware/API
    app._recorder = MagicMock()
    app._inserter = MagicMock()
    app._transcriber = MagicMock()
    app._engine = None
    app._audio_cache = None
    app._ui = MagicMock()
    app._ui._root = MagicMock()

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecordingTimerCreated(unittest.TestCase):
    """Timer is created when max_recording_seconds > 0."""

    def test_timer_created_on_start(self):
        app = _make_app(max_recording_seconds=300)

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.start.return_value = None

        with patch("src.app.AudioRecorder", return_value=mock_recorder_instance):
            with patch("threading.Timer") as mock_timer_cls:
                mock_timer = MagicMock()
                mock_timer_cls.return_value = mock_timer

                app._start_recording()

                mock_timer_cls.assert_called_once_with(300, app._auto_stop_recording)
                mock_timer.start.assert_called_once()
                self.assertTrue(mock_timer.daemon)

    def tearDown(self):
        # Cancel timer if one was created
        if hasattr(app := getattr(self, '_app', None), '_recording_timer') and app._recording_timer:
            app._recording_timer.cancel()


class TestNoTimerWhenZero(unittest.TestCase):
    """Timer is NOT created when max_recording_seconds = 0."""

    def test_no_timer_when_zero(self):
        app = _make_app(max_recording_seconds=0)

        mock_recorder_instance = MagicMock()

        with patch("src.app.AudioRecorder", return_value=mock_recorder_instance):
            with patch("threading.Timer") as mock_timer_cls:
                app._start_recording()

                mock_timer_cls.assert_not_called()
                self.assertIsNone(app._recording_timer)


class TestTimerCancelledOnManualStop(unittest.TestCase):
    """Timer is cancelled when user manually stops recording."""

    def test_timer_cancelled_on_stop(self):
        app = _make_app(max_recording_seconds=300)

        # Simulate recording in progress with a live timer
        mock_timer = MagicMock()
        app._recording_timer = mock_timer
        app._recording = True
        app._recorder = MagicMock()
        app._processing = False

        app._stop_recording_async()

        mock_timer.cancel.assert_called_once()
        self.assertIsNone(app._recording_timer)
        self.assertFalse(app._recording)


class TestTimerCancelledOnCancel(unittest.TestCase):
    """Timer is cancelled when user cancels recording via X button."""

    def test_timer_cancelled_on_cancel(self):
        app = _make_app(max_recording_seconds=300)

        mock_timer = MagicMock()
        app._recording_timer = mock_timer
        app._recording = True
        app._recorder = MagicMock()

        app.on_cancel_recording()

        mock_timer.cancel.assert_called_once()
        self.assertIsNone(app._recording_timer)
        self.assertFalse(app._recording)


class TestAutoStopRecording(unittest.TestCase):
    """_auto_stop_recording() triggers stop when recording is active."""

    def test_auto_stop_calls_stop_async(self):
        app = _make_app(max_recording_seconds=5)
        app._recording = True
        app._recorder = MagicMock()
        app._processing = False

        with patch.object(app, "_stop_recording_async") as mock_stop:
            app._auto_stop_recording()
            mock_stop.assert_called_once()

    def test_auto_stop_noop_when_not_recording(self):
        """If recording already stopped, _auto_stop_recording() does nothing."""
        app = _make_app(max_recording_seconds=5)
        app._recording = False

        with patch.object(app, "_stop_recording_async") as mock_stop:
            app._auto_stop_recording()
            mock_stop.assert_not_called()


class TestCountdownDisplay(unittest.TestCase):
    """_update_recording_countdown() calls update_countdown on UI."""

    def test_countdown_updates_ui(self):
        app = _make_app(max_recording_seconds=300)
        app._recording = True
        app._recording_start_time = time.time() - 10  # 10 seconds elapsed → 290 remaining

        mock_root = MagicMock()
        app._ui._root = mock_root

        app._update_recording_countdown()

        # Should call update_countdown with ~290 remaining
        app._ui.update_countdown.assert_called_once()
        remaining = app._ui.update_countdown.call_args[0][0]
        self.assertGreaterEqual(remaining, 288)  # allow 2 second slack
        self.assertLessEqual(remaining, 291)

    def test_countdown_clears_when_stopped(self):
        app = _make_app(max_recording_seconds=300)
        app._recording = False

        app._update_recording_countdown()

        app._ui.update_countdown.assert_called_once_with(None)

    def test_no_countdown_when_unlimited(self):
        """When max_recording_seconds = 0, countdown loop does nothing."""
        app = _make_app(max_recording_seconds=0)
        app._recording = True
        app._recording_start_time = time.time()

        app._update_recording_countdown()

        app._ui.update_countdown.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
