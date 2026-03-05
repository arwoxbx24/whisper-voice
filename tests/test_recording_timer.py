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

from __future__ import annotations

import os
import sys
import types
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub ALL heavy/hardware dependencies before any src import
# ---------------------------------------------------------------------------

# 1. sounddevice (requires PortAudio)
if "sounddevice" not in sys.modules:
    sd_mod = types.ModuleType("sounddevice")

    class _FakeInputStream:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def stop(self): pass
        def close(self): pass

    sd_mod.InputStream = _FakeInputStream
    sd_mod.sleep = lambda ms: None
    sys.modules["sounddevice"] = sd_mod

# 2. pynput (requires X11/display)
if "pynput" not in sys.modules:
    _kb_mod = MagicMock()
    _kb_mod.Key = MagicMock()
    _kb_mod.Key.space = MagicMock(name="space")
    _kb_mod.KeyCode = MagicMock()
    _kb_mod.KeyCode.from_char = lambda c: MagicMock()
    _kb_mod.HotKey = MagicMock()
    _kb_mod.HotKey.parse = lambda combo: []
    _kb_mod.Listener = MagicMock()

    _mouse_mod = MagicMock()
    _mouse_mod.Button = MagicMock()
    _mouse_mod.Listener = MagicMock()

    _pynput_mod = MagicMock()
    _pynput_mod.keyboard = _kb_mod
    _pynput_mod.mouse = _mouse_mod

    sys.modules["pynput"] = _pynput_mod
    sys.modules["pynput.keyboard"] = _kb_mod
    sys.modules["pynput.mouse"] = _mouse_mod

# 3. PIL (requires Pillow)
if "PIL" not in sys.modules:
    _pil_mod = types.ModuleType("PIL")
    _image_mod = types.ModuleType("PIL.Image")
    _image_mod.new = lambda *a, **kw: MagicMock()
    _draw_mod = types.ModuleType("PIL.ImageDraw")
    _draw_mod.Draw = lambda img: MagicMock()
    _font_mod = types.ModuleType("PIL.ImageFont")
    _pil_mod.Image = _image_mod
    _pil_mod.ImageDraw = _draw_mod
    _pil_mod.ImageFont = _font_mod
    sys.modules["PIL"] = _pil_mod
    sys.modules["PIL.Image"] = _image_mod
    sys.modules["PIL.ImageDraw"] = _draw_mod
    sys.modules["PIL.ImageFont"] = _font_mod

# 4. tkinter (requires display)
if "tkinter" not in sys.modules:
    _tk_mod = types.ModuleType("tkinter")
    _tk_mod.Tk = MagicMock
    _tk_mod.Toplevel = MagicMock
    _tk_mod.Canvas = MagicMock
    _tk_mod.Button = MagicMock
    _tk_mod.Label = MagicMock
    _tk_mod.StringVar = MagicMock
    _tkfont_mod = types.ModuleType("tkinter.font")
    sys.modules["tkinter"] = _tk_mod
    sys.modules["tkinter.font"] = _tkfont_mod

# 5. pystray (requires tray support)
if "pystray" not in sys.modules:
    _pystray_mod = types.ModuleType("pystray")
    _pystray_mod.Icon = MagicMock
    _pystray_mod.MenuItem = MagicMock
    _menu_mock = MagicMock()
    _menu_mock.SEPARATOR = MagicMock()
    _pystray_mod.Menu = _menu_mock
    sys.modules["pystray"] = _pystray_mod

# Now it is safe to import src modules
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.app import WhisperVoiceApp  # noqa: E402


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_app(max_recording_seconds: int = 300) -> WhisperVoiceApp:
    """Create a WhisperVoiceApp with mocked dependencies."""
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

    # Mock all sub-components to avoid real hardware / API calls
    app._recorder = MagicMock()
    app._inserter = MagicMock()
    app._transcriber = MagicMock()
    app._engine = None
    app._audio_cache = None
    app._ui = MagicMock()
    app._ui._root = MagicMock()
    app._recording = False
    app._processing = False

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecordingTimerCreated(unittest.TestCase):
    """Timer is created when max_recording_seconds > 0."""

    def test_timer_created_on_start(self):
        app = _make_app(max_recording_seconds=300)
        mock_recorder_instance = MagicMock()

        with patch("src.app.AudioRecorder", return_value=mock_recorder_instance):
            with patch("threading.Timer") as mock_timer_cls:
                mock_timer = MagicMock()
                mock_timer_cls.return_value = mock_timer

                app._start_recording()

                mock_timer_cls.assert_called_once_with(300, app._auto_stop_recording)
                mock_timer.start.assert_called_once()
                self.assertEqual(app._recording_timer, mock_timer)

        app._recording_timer = None  # cleanup


class TestNoTimerWhenZero(unittest.TestCase):
    """Timer is NOT created when max_recording_seconds = 0 (unlimited)."""

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
    """_auto_stop_recording() behaviour."""

    def test_auto_stop_calls_stop_async(self):
        """When recording is active, _auto_stop_recording calls _stop_recording_async."""
        app = _make_app(max_recording_seconds=5)
        app._recording = True
        app._recorder = MagicMock()
        app._processing = False

        with patch.object(app, "_stop_recording_async") as mock_stop:
            app._auto_stop_recording()
            mock_stop.assert_called_once()

    def test_auto_stop_noop_when_not_recording(self):
        """If recording already stopped, _auto_stop_recording does nothing."""
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
        app._recording_start_time = time.time() - 10  # 10 s elapsed → ~290 remaining

        app._update_recording_countdown()

        app._ui.update_countdown.assert_called_once()
        remaining = app._ui.update_countdown.call_args[0][0]
        self.assertGreaterEqual(remaining, 288, "Should be ~290 with 2 s slack")
        self.assertLessEqual(remaining, 291)

    def test_countdown_clears_when_stopped(self):
        """When not recording, countdown is cleared (None)."""
        app = _make_app(max_recording_seconds=300)
        app._recording = False

        app._update_recording_countdown()

        app._ui.update_countdown.assert_called_once_with(None)

    def test_no_countdown_when_unlimited(self):
        """When max_recording_seconds = 0, no countdown is displayed."""
        app = _make_app(max_recording_seconds=0)
        app._recording = True
        app._recording_start_time = time.time()

        app._update_recording_countdown()

        app._ui.update_countdown.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
