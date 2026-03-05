"""Tests for src/audio_recorder.py — AudioRecorder without real hardware.

sounddevice imports PortAudio at module load time and will fail on CI.
We patch it before importing the module.
"""
import os
import sys
import threading
import wave
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Patch sounddevice before importing the module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_sounddevice():
    """Patch sounddevice globally so PortAudio is never loaded."""
    mock_sd = MagicMock()
    # Ensure sd.InputStream context manager works
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value.__enter__ = lambda s: mock_stream
    mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
    with patch.dict(sys.modules, {"sounddevice": mock_sd}):
        # Remove cached module if already loaded
        if "src.audio_recorder" in sys.modules:
            del sys.modules["src.audio_recorder"]
        yield mock_sd
    # Clean up module cache after test
    if "src.audio_recorder" in sys.modules:
        del sys.modules["src.audio_recorder"]


def get_recorder(**kwargs):
    """Import and instantiate AudioRecorder with mocked sounddevice."""
    from src.audio_recorder import AudioRecorder
    return AudioRecorder(**kwargs)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestAudioRecorderInit:
    def test_default_parameters(self, mock_sounddevice):
        r = get_recorder()
        assert r.sample_rate == 16000
        assert r.channels == 1
        assert r.dtype == "int16"
        assert r.level_callback is None

    def test_custom_parameters(self, mock_sounddevice):
        cb = lambda x: x
        r = get_recorder(sample_rate=44100, channels=2, dtype="float32", level_callback=cb)
        assert r.sample_rate == 44100
        assert r.channels == 2
        assert r.dtype == "float32"
        assert r.level_callback is cb

    def test_initial_state_not_recording(self, mock_sounddevice):
        r = get_recorder()
        assert not r.is_recording()

    def test_initial_audio_level_zero(self, mock_sounddevice):
        r = get_recorder()
        assert r.get_audio_level() == 0.0


# ---------------------------------------------------------------------------
# Start / Stop guards
# ---------------------------------------------------------------------------

class TestAudioRecorderStartStop:
    def test_double_start_raises(self, mock_sounddevice):
        r = get_recorder()
        # Manually set recording to simulate in-progress
        with r._lock:
            r._recording = True
        with pytest.raises(RuntimeError, match="already in progress"):
            r.start()
        with r._lock:
            r._recording = False

    def test_stop_without_start_raises(self, mock_sounddevice):
        r = get_recorder()
        with pytest.raises(RuntimeError, match="No recording in progress"):
            r.stop()


# ---------------------------------------------------------------------------
# _save_wav
# ---------------------------------------------------------------------------

class TestSaveWav:
    def test_save_empty_frames_creates_valid_wav(self, mock_sounddevice):
        r = get_recorder()
        path = r._save_wav([])
        try:
            assert os.path.exists(path)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
        finally:
            os.unlink(path)

    def test_save_frames_creates_valid_wav_with_data(self, mock_sounddevice):
        r = get_recorder()
        frames = [np.zeros((800, 1), dtype=np.int16) for _ in range(3)]
        path = r._save_wav(frames)
        try:
            assert os.path.exists(path)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                nframes = wf.getnframes()
                assert nframes > 0
        finally:
            os.unlink(path)

    def test_save_uses_whisper_prefix(self, mock_sounddevice):
        r = get_recorder()
        path = r._save_wav([])
        try:
            assert "whisper_rec_" in os.path.basename(path)
            assert path.endswith(".wav")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Thread-safe getters
# ---------------------------------------------------------------------------

class TestGetAudioLevel:
    def test_get_level_thread_safe(self, mock_sounddevice):
        r = get_recorder()
        with r._lock:
            r._current_level = 0.75
        assert r.get_audio_level() == 0.75

    def test_is_recording_reflects_state(self, mock_sounddevice):
        r = get_recorder()
        assert not r.is_recording()
        with r._lock:
            r._recording = True
        assert r.is_recording()
        with r._lock:
            r._recording = False
