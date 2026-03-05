"""
Audio recorder module for Whisper voice transcription.

Records microphone input in WAV format (16kHz, mono, 16-bit) —
the optimal format for OpenAI Whisper models.

Thread-safe: recording runs in a background thread.
"""

import sounddevice as sd
import numpy as np
import wave
import tempfile
import threading
import os
from typing import Callable, Optional


class AudioRecorder:
    """
    Records audio from the system microphone.

    Usage:
        recorder = AudioRecorder()
        recorder.start()
        # ... do stuff while recording ...
        wav_path = recorder.stop()  # returns path to saved WAV file
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        level_callback: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            sample_rate: Sample rate in Hz. 16000 is optimal for Whisper.
            channels: Number of channels. 1 (mono) is required by Whisper.
            dtype: Sample format. 'int16' = 16-bit PCM.
            level_callback: Optional callable(level: float) called on each
                            audio chunk with the current RMS level (0.0–1.0).
                            Useful for driving a UI volume indicator.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.level_callback = level_callback

        self._lock = threading.Lock()
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._current_level: float = 0.0
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._stream_ready = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start recording audio in a background thread.

        Raises:
            RuntimeError: if recording is already in progress.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress. Call stop() first.")
            self._recording = True
            self._frames = []
            self._current_level = 0.0

        self._stream_ready.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> str:
        """Stop recording and save audio to a temporary WAV file.

        Returns:
            Absolute path to the saved WAV file. The caller is responsible
            for deleting the file when done.

        Raises:
            RuntimeError: if no recording is in progress.
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("No recording in progress. Call start() first.")
            self._recording = False

        # Wait for the stream to be assigned by _record_loop before accessing it
        self._stream_ready.wait(timeout=2.0)

        # Signal the stream to stop and wait for the thread to finish
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        with self._lock:
            frames = list(self._frames)

        return self._save_wav(frames)

    def get_audio_level(self) -> float:
        """Return the current audio level for UI indicators.

        Returns:
            Float in [0.0, 1.0] representing the RMS level of the most
            recent audio chunk. Returns 0.0 when not recording.
        """
        with self._lock:
            return self._current_level

    def is_recording(self) -> bool:
        """Return True if recording is currently active."""
        with self._lock:
            return self._recording

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_loop(self) -> None:
        """Background thread: opens a sounddevice InputStream and collects chunks."""
        blocksize = int(self.sample_rate * 0.05)  # 50 ms chunks

        def _callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ARG001
            """sounddevice callback — called from the audio thread."""
            if status:
                # Non-fatal: log but keep recording
                pass

            chunk = indata.copy()

            # Compute RMS level, normalise to 0.0–1.0 for int16 range
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            level = min(rms / 32768.0, 1.0)

            with self._lock:
                if not self._recording:
                    return
                self._frames.append(chunk)
                self._current_level = level

            if self.level_callback is not None:
                try:
                    self.level_callback(level)
                except Exception:
                    pass  # Never crash the audio thread due to callback errors

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=blocksize,
                callback=_callback,
            ) as stream:
                self._stream = stream
                self._stream_ready.set()
                # Block until _recording is False
                while True:
                    with self._lock:
                        if not self._recording:
                            break
                    sd.sleep(10)  # sleep 10 ms, then re-check
        except Exception as exc:
            self._stream_ready.set()  # unblock stop() if waiting
            with self._lock:
                self._recording = False
            raise RuntimeError(f"Audio stream error: {exc}") from exc
        finally:
            with self._lock:
                self._current_level = 0.0

    def _save_wav(self, frames: list[np.ndarray]) -> str:
        """Concatenate frames and write a WAV file in a temp directory.

        Returns:
            Path to the written file.
        """
        # Use delete=False so the caller can open the file after this method returns
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="whisper_rec_",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()

        if not frames:
            # Write an empty-but-valid WAV file so downstream code doesn't crash
            audio_data = np.zeros(0, dtype=np.int16)
        else:
            audio_data = np.concatenate(frames, axis=0)

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        return tmp_path


# ------------------------------------------------------------------
# Quick smoke-test (run directly: python audio_recorder.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Recording for 3 seconds...")
    recorder = AudioRecorder(level_callback=lambda lvl: print(f"  level: {lvl:.3f}"))
    recorder.start()
    time.sleep(3)
    path = recorder.stop()
    size = os.path.getsize(path)
    print(f"Saved: {path}  ({size} bytes)")
    print("Done.")
