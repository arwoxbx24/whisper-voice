"""Tests for SoundFeedback."""
import sys
import threading
import unittest
from unittest.mock import MagicMock, call, patch


# Provide a stub for winsound on non-Windows platforms
if "winsound" not in sys.modules:
    sys.modules["winsound"] = MagicMock()

from src.sound_feedback import SoundFeedback  # noqa: E402


class TestSoundFeedbackDisabled(unittest.TestCase):
    """When enabled=False no sound calls should be made."""

    def setUp(self):
        self.sf = SoundFeedback(enabled=False)

    def test_start_recording_no_sound(self):
        with patch.object(self.sf, "_play_async") as mock_play:
            self.sf.play_start_recording()
            mock_play.assert_not_called()

    def test_stop_recording_no_sound(self):
        with patch.object(self.sf, "_play_async") as mock_play:
            self.sf.play_stop_recording()
            mock_play.assert_not_called()

    def test_transcription_complete_no_sound(self):
        with patch.object(self.sf, "_play_async_sequence") as mock_seq:
            self.sf.play_transcription_complete()
            mock_seq.assert_not_called()


class TestSoundFeedbackEnabled(unittest.TestCase):
    """When enabled=True the correct sounds should be triggered."""

    def setUp(self):
        self.sf = SoundFeedback(enabled=True)

    def test_start_recording_sound(self):
        with patch.object(self.sf, "_play_async") as mock_play:
            self.sf.play_start_recording()
            mock_play.assert_called_once_with(440, 100)

    def test_stop_recording_sound(self):
        with patch.object(self.sf, "_play_async") as mock_play:
            self.sf.play_stop_recording()
            mock_play.assert_called_once_with(880, 100)

    def test_transcription_complete_sound(self):
        with patch.object(self.sf, "_play_async_sequence") as mock_seq:
            self.sf.play_transcription_complete()
            mock_seq.assert_called_once_with([(660, 80), (880, 80)])

    def test_start_stop_different_frequencies(self):
        """Start and stop must use different frequencies."""
        start_freq = None
        stop_freq = None

        with patch.object(self.sf, "_play_async") as mock_play:
            self.sf.play_start_recording()
            start_freq = mock_play.call_args[0][0]
            mock_play.reset_mock()

            self.sf.play_stop_recording()
            stop_freq = mock_play.call_args[0][0]

        self.assertNotEqual(start_freq, stop_freq)


class TestSoundFeedbackPlatformWindows(unittest.TestCase):
    """On Windows platform winsound.Beep should be called."""

    def test_windows_plays_beep(self):
        with patch("platform.system", return_value="Windows"):
            sf = SoundFeedback(enabled=True)
        mock_winsound = MagicMock()
        with patch.dict("sys.modules", {"winsound": mock_winsound}):
            sf._play_sound(440, 100)
            mock_winsound.Beep.assert_called_once_with(440, 100)


class TestSoundFeedbackPlatformLinux(unittest.TestCase):
    """On Linux platform subprocess tools or terminal bell should be used."""

    def setUp(self):
        with patch("platform.system", return_value="Linux"):
            self.sf = SoundFeedback(enabled=True)

    def test_linux_falls_back_to_bell_when_no_tools(self):
        """When sox and beep are missing, print('\\a') should be called."""
        import subprocess

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("builtins.print") as mock_print:
                self.sf._play_sound(440, 100)
                mock_print.assert_called_once_with("\a", end="", flush=True)

    def test_linux_uses_sox_when_available(self):
        import subprocess

        with patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            self.sf._play_sound(440, 100)
            # First call should be to sox
            first_call_cmd = mock_run.call_args_list[0][0][0]
            self.assertEqual(first_call_cmd[0], "sox")


class TestSoundFeedbackAsync(unittest.TestCase):
    """Async helpers must spawn daemon threads."""

    def setUp(self):
        self.sf = SoundFeedback(enabled=True)

    def test_play_async_creates_daemon_thread(self):
        created_threads = []
        original_thread_class = threading.Thread

        def capture_thread(*args, **kwargs):
            t = original_thread_class(*args, **kwargs)
            created_threads.append(t)
            return t

        with patch("src.sound_feedback.threading.Thread", side_effect=capture_thread):
            with patch.object(self.sf, "_play_sound"):
                self.sf._play_async(440, 100)

        self.assertTrue(len(created_threads) >= 1)
        self.assertTrue(created_threads[0].daemon)

    def test_play_async_sequence_creates_daemon_thread(self):
        created_threads = []
        original_thread_class = threading.Thread

        def capture_thread(*args, **kwargs):
            t = original_thread_class(*args, **kwargs)
            created_threads.append(t)
            return t

        with patch("src.sound_feedback.threading.Thread", side_effect=capture_thread):
            self.sf._play_async_sequence([(440, 100)])

        self.assertTrue(len(created_threads) >= 1)
        self.assertTrue(created_threads[0].daemon)


if __name__ == "__main__":
    unittest.main()
