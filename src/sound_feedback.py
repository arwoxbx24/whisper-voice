"""Sound feedback for recording events."""
import logging
import platform
import threading
import time

logger = logging.getLogger(__name__)


class SoundFeedback:
    """Plays short sounds on recording events."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._is_windows = platform.system() == "Windows"

    def play_start_recording(self) -> None:
        """Short ascending tone — recording started."""
        if not self.enabled:
            return
        self._play_async(440, 100)  # A4, 100ms

    def play_stop_recording(self) -> None:
        """Short descending tone — recording stopped."""
        if not self.enabled:
            return
        self._play_async(880, 100)  # A5, 100ms

    def play_transcription_complete(self) -> None:
        """Double beep — transcription done, text inserted."""
        if not self.enabled:
            return
        self._play_async_sequence([(660, 80), (880, 80)])  # Two tones

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_async(self, frequency: int, duration_ms: int) -> None:
        """Play sound in daemon thread to avoid blocking."""
        t = threading.Thread(
            target=self._play_sound,
            args=(frequency, duration_ms),
            daemon=True,
        )
        t.start()

    def _play_async_sequence(self, tones: list) -> None:
        """Play sequence of (freq, dur) in daemon thread."""

        def _seq() -> None:
            for freq, dur in tones:
                self._play_sound(freq, dur)
                time.sleep(0.05)  # Small gap between tones

        t = threading.Thread(target=_seq, daemon=True)
        t.start()

    def _play_sound(self, frequency: int, duration_ms: int) -> None:
        """Platform-specific sound playback."""
        try:
            if self._is_windows:
                import winsound  # noqa: PLC0415 — Windows-only, import inside method

                winsound.Beep(frequency, duration_ms)
            else:
                # Linux/macOS: try paplay, fall back to terminal bell
                import subprocess

                try:
                    # Generate a raw PCM tone via sox if available
                    subprocess.run(
                        [
                            "sox",
                            "-n",
                            "-t",
                            "alsa",
                            "default",
                            "synth",
                            str(duration_ms / 1000),
                            "sine",
                            str(frequency),
                        ],
                        timeout=1,
                        capture_output=True,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    try:
                        # Fallback: beep command
                        subprocess.run(
                            ["beep", "-f", str(frequency), "-l", str(duration_ms)],
                            timeout=1,
                            capture_output=True,
                        )
                    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                        # Final fallback: terminal bell
                        print("\a", end="", flush=True)
        except Exception as exc:
            logger.debug("Sound playback failed: %s", exc)
