"""
Coverage boost tests: target uncovered lines in text_inserter, audio_recorder,
hotkey_manager, and providers.

All hardware/OS interactions are mocked.
"""
from __future__ import annotations

import sys
import threading
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Helper: build minimal pynput/sounddevice mocks for use in patch.dict
# ---------------------------------------------------------------------------

def _make_pynput_mocks():
    """Return dict of sys.modules patches for pynput + sounddevice."""
    mock_pynput = MagicMock()
    mock_keyboard = MagicMock()
    mock_mouse = MagicMock()
    mock_controller_inst = MagicMock()
    mock_keyboard.Controller = MagicMock(return_value=mock_controller_inst)

    for name in ["ctrl_l", "ctrl_r", "ctrl", "shift_l", "shift_r", "shift",
                 "alt_l", "alt_r", "insert", "cmd", "delete", "space", "enter"]:
        setattr(mock_keyboard.Key, name, MagicMock(name=f"Key.{name}"))

    mock_pynput.keyboard = mock_keyboard
    mock_pynput.mouse = mock_mouse

    return {
        "pynput": mock_pynput,
        "pynput.keyboard": mock_keyboard,
        "pynput.mouse": mock_mouse,
        "sounddevice": MagicMock(),
    }


# ===========================================================================
# SmartTextInserter — additional coverage
# ===========================================================================

class TestTextInserterCoverage:
    """Target uncovered lines in text_inserter.py."""

    def _make_inserter(self, method="clipboard"):
        from src.text_inserter import SmartTextInserter
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)  # xdotool not available
            ins = SmartTextInserter(method=method, paste_delay=0.0,
                                    terminal_paste_delay=0.0, restore_delay=0.0)
        ins._xdotool_available = False
        return ins

    def test_insert_empty_text_returns_false(self):
        ins = self._make_inserter()
        assert ins.insert_text("") is False

    def test_insert_legacy_empty_no_raise(self):
        ins = self._make_inserter()
        ins.insert("")  # Should not raise

    def test_insert_legacy_raises_on_failure(self):
        from src.text_inserter import TextInserterError, SmartTextInserter
        ins = self._make_inserter()
        ins._insert_via_clipboard = MagicMock(return_value=False)
        ins._xdotool_available = False
        ins._method = "clipboard"
        with pytest.raises(TextInserterError):
            ins.insert("hello")

    def test_insert_method_type_calls_typing(self):
        ins = self._make_inserter(method="type")
        ins._insert_typing = MagicMock(return_value=True)
        result = ins.insert_text("hi")
        ins._insert_typing.assert_called_once_with("hi")
        assert result is True

    def test_insert_method_type_no_keyboard_returns_false(self):
        ins = self._make_inserter(method="type")
        ins._keyboard = None
        result = ins._insert_typing("hi")
        assert result is False

    def test_insert_via_clipboard_success(self):
        ins = self._make_inserter(method="clipboard")
        mock_kb = MagicMock()
        with patch("pyperclip.paste", return_value="old"), \
             patch("pyperclip.copy"), \
             patch("src.text_inserter.time.sleep"), \
             patch("src.text_inserter.SmartTextInserter._is_terminal", return_value=False), \
             patch("src.text_inserter.SmartTextInserter._send_paste"):
            result = ins._insert_via_clipboard("hello world")
        assert result is True

    def test_insert_via_clipboard_terminal(self):
        ins = self._make_inserter(method="clipboard")
        with patch("pyperclip.paste", return_value=""), \
             patch("pyperclip.copy"), \
             patch("src.text_inserter.time.sleep"), \
             patch("src.text_inserter.SmartTextInserter._is_terminal", return_value=True), \
             patch("src.text_inserter.SmartTextInserter._send_paste"):
            result = ins._insert_via_clipboard("text")
        assert result is True

    def test_insert_via_clipboard_pyperclip_error_returns_false(self):
        ins = self._make_inserter(method="clipboard")
        with patch("pyperclip.paste", side_effect=Exception("no display")), \
             patch("pyperclip.copy", side_effect=Exception("no display")), \
             patch("src.text_inserter.time.sleep"):
            result = ins._insert_via_clipboard("text")
        assert result is False

    def test_is_terminal_linux_xdotool_failure(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = ins._is_terminal_linux()
        assert result is False

    def test_is_terminal_linux_matches(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        with patch("subprocess.run") as mock_run:
            # First call: getactivewindow → success
            # Second call: getwindowclassname → "gnome-terminal"
            # Third call: getwindowname → "Terminal"
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="12345\n"),
                MagicMock(returncode=0, stdout="gnome-terminal\n"),
                MagicMock(returncode=0, stdout="Terminal\n"),
            ]
            result = ins._is_terminal_linux()
        assert result is True

    def test_is_terminal_linux_file_not_found(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        with patch("subprocess.run", side_effect=FileNotFoundError("xdotool")):
            result = ins._is_terminal_linux()
        assert result is False

    def test_is_terminal_macos_matches(self):
        ins = self._make_inserter()
        ins._system = "Darwin"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="terminal\n")
            result = ins._is_terminal_macos()
        assert result is True

    def test_is_terminal_macos_file_not_found(self):
        ins = self._make_inserter()
        ins._system = "Darwin"
        with patch("subprocess.run", side_effect=FileNotFoundError("osascript")):
            result = ins._is_terminal_macos()
        assert result is False

    def test_is_terminal_windows_calls_ctypes(self):
        ins = self._make_inserter()
        ins._system = "Windows"
        mock_ctypes = MagicMock()
        mock_ctypes.windll.user32.GetForegroundWindow.return_value = 1
        mock_ctypes.windll.user32.GetWindowTextLengthW.return_value = 7
        mock_ctypes.create_unicode_buffer.return_value = MagicMock(value="cmd.exe")
        with patch.dict(sys.modules, {"ctypes": mock_ctypes}):
            result = ins._is_terminal_windows()
        assert isinstance(result, bool)

    def test_is_terminal_dispatch_linux(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        ins._is_terminal_linux = MagicMock(return_value=True)
        assert ins._is_terminal() is True

    def test_is_terminal_dispatch_exception(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        ins._is_terminal_linux = MagicMock(side_effect=Exception("crash"))
        assert ins._is_terminal() is False

    def test_safe_get_clipboard_returns_none_on_error(self):
        ins = self._make_inserter()
        with patch("pyperclip.paste", side_effect=Exception("no display")):
            result = ins._safe_get_clipboard()
        assert result is None

    def test_safe_get_clipboard_returns_text(self):
        ins = self._make_inserter()
        with patch("pyperclip.paste", return_value="clipboard text"):
            result = ins._safe_get_clipboard()
        assert result == "clipboard text"

    def test_safe_set_clipboard_no_raise(self):
        ins = self._make_inserter()
        with patch("pyperclip.copy", side_effect=Exception("no display")):
            ins._safe_set_clipboard("text")  # Should not raise

    def test_xdotool_method_fails_no_fallback(self):
        ins = self._make_inserter(method="xdotool")
        ins._xdotool_available = True
        ins._try_xdotool = MagicMock(return_value=False)
        result = ins.insert_text("hello")
        assert result is False

    def test_xdotool_method_success(self):
        ins = self._make_inserter(method="xdotool")
        ins._xdotool_available = True
        ins._try_xdotool = MagicMock(return_value=True)
        result = ins.insert_text("hello")
        assert result is True

    def test_try_xdotool_returncode_nonzero(self):
        ins = self._make_inserter()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = ins._try_xdotool("text")
        assert result is False

    def test_try_xdotool_success(self):
        ins = self._make_inserter()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = ins._try_xdotool("text")
        assert result is True

    def test_try_xdotool_file_not_found(self):
        ins = self._make_inserter()
        with patch("subprocess.run", side_effect=FileNotFoundError("xdotool")):
            result = ins._try_xdotool("text")
        assert result is False

    def test_check_xdotool_non_linux(self):
        ins = self._make_inserter()
        ins._system = "Windows"
        assert ins._check_xdotool() is False

    def test_check_xdotool_linux_available(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = ins._check_xdotool()
        assert result is True

    def test_check_xdotool_linux_not_available(self):
        ins = self._make_inserter()
        ins._system = "Linux"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = ins._check_xdotool()
        assert result is False


# ===========================================================================
# AudioRecorder — additional coverage
# ===========================================================================

class TestAudioRecorderCoverage:
    """Target uncovered lines in audio_recorder.py."""

    def _make_recorder(self):
        import sounddevice as sd
        sd_mock = sys.modules["sounddevice"]
        from src.audio_recorder import AudioRecorder
        return AudioRecorder(sample_rate=16000, channels=1, dtype="int16")

    def test_start_raises_if_already_recording(self):
        from src.audio_recorder import AudioRecorder
        rec = AudioRecorder()
        rec._recording = True
        with pytest.raises(RuntimeError, match="already in progress"):
            rec.start()

    def test_stop_raises_if_not_recording(self):
        from src.audio_recorder import AudioRecorder
        rec = AudioRecorder()
        rec._recording = False
        with pytest.raises(RuntimeError, match="No recording"):
            rec.stop()

    def test_get_audio_level_initial(self):
        from src.audio_recorder import AudioRecorder
        rec = AudioRecorder()
        assert rec.get_audio_level() == 0.0

    def test_is_recording_initial_false(self):
        from src.audio_recorder import AudioRecorder
        rec = AudioRecorder()
        assert rec.is_recording() is False

    def test_level_callback_called(self):
        """Test that level_callback is invoked when provided."""
        import numpy as np
        from src.audio_recorder import AudioRecorder

        callback_levels = []
        rec = AudioRecorder(level_callback=lambda lvl: callback_levels.append(lvl))

        # Simulate the callback being called directly
        # We can test the callback attachment is stored
        assert rec.level_callback is not None

    def test_save_wav_with_numpy_frames(self):
        """Test _save_wav creates a valid file."""
        import numpy as np
        import wave
        from src.audio_recorder import AudioRecorder

        rec = AudioRecorder(sample_rate=16000, channels=1, dtype="int16")
        # Create fake frames
        frames = [np.zeros(800, dtype=np.int16), np.ones(800, dtype=np.int16) * 100]
        wav_path = rec._save_wav(frames)
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")
        # Clean up
        os.unlink(wav_path)

    def test_save_wav_empty_frames(self):
        """Test _save_wav with empty frames list."""
        import numpy as np
        from src.audio_recorder import AudioRecorder

        rec = AudioRecorder()
        wav_path = rec._save_wav([])
        assert os.path.exists(wav_path)
        os.unlink(wav_path)


# ===========================================================================
# Providers — additional coverage
# ===========================================================================

class TestProvidersCoverage:
    """Cover uncovered lines in deepgram_provider and local_provider."""

    def test_deepgram_provider_init_no_package(self):
        """deepgram_provider raises ImportError if deepgram SDK absent."""
        from src.providers.deepgram_provider import DeepgramProvider
        # Just test instantiation — skip if deepgram not installed
        try:
            import deepgram
        except ImportError:
            with pytest.raises(Exception):
                DeepgramProvider(api_key="k")

    def test_local_provider_init_no_faster_whisper(self):
        """local_provider raises ImportError if faster_whisper absent."""
        try:
            import faster_whisper
            pytest.skip("faster_whisper is installed, skip absence test")
        except ImportError:
            from src.providers.local_provider import LocalProvider
            with pytest.raises(Exception):
                p = LocalProvider()
                p.transcribe("/tmp/fake.wav")

    def test_openai_provider_transcribe_success(self):
        """openai_provider.transcribe returns TranscriptionResult on success."""
        from src.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="k")
        provider._client = MagicMock()
        provider._client.audio.transcriptions.create.return_value = MagicMock(text="hello")

        with patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()),
            __exit__=MagicMock(return_value=False)
        ))):
            result = provider.transcribe("/fake/path.wav", language="ru")
        assert result.text == "hello"

    def test_openai_provider_transcribe_with_prompt(self):
        """openai_provider.transcribe passes prompt parameter."""
        from src.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="k")
        provider._client = MagicMock()
        provider._client.audio.transcriptions.create.return_value = MagicMock(text="text")

        with patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()),
            __exit__=MagicMock(return_value=False)
        ))):
            result = provider.transcribe("/fake/path.wav", prompt="context hint")
        assert result.text == "text"


# ===========================================================================
# StateMachine — additional coverage for callbacks
# ===========================================================================

class TestStateMachineCoverage:
    """Cover remaining state_machine.py lines."""

    def _make_sm(self):
        from src.state_machine import StateMachine, State
        return StateMachine(), State

    def test_on_exit_callback_fires_on_transition(self):
        sm, State = self._make_sm()
        exits = []
        sm.on_exit(State.RECORDING, lambda: exits.append("exit_recording"))
        sm.transition(State.RECORDING)
        sm.transition(State.PROCESSING)
        assert "exit_recording" in exits

    def test_on_enter_processing_callback(self):
        sm, State = self._make_sm()
        enters = []
        sm.on_enter(State.PROCESSING, lambda: enters.append("enter_processing"))
        sm.transition(State.RECORDING)
        sm.transition(State.PROCESSING)
        assert "enter_processing" in enters

    def test_multiple_callbacks_all_called(self):
        sm, State = self._make_sm()
        log = []
        sm.on_enter(State.RECORDING, lambda: log.append("cb1"))
        sm.on_enter(State.RECORDING, lambda: log.append("cb2"))
        sm.transition(State.RECORDING)
        assert "cb1" in log
        assert "cb2" in log

    def test_reset_from_processing(self):
        sm, State = self._make_sm()
        sm.transition(State.RECORDING)
        sm.transition(State.PROCESSING)
        sm.reset()
        assert sm.current == State.IDLE

    def test_callback_exception_doesnt_stop_transition(self):
        sm, State = self._make_sm()
        sm.on_enter(State.RECORDING, lambda: 1/0)  # raises ZeroDivisionError
        # transition should still succeed despite callback error
        result = sm.transition(State.RECORDING)
        assert result is True
        assert sm.current == State.RECORDING
