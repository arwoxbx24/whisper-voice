import time
import logging
import threading
from typing import Optional

from .base import STTProvider, TranscriptionResult, TransientError, PermanentError

logger = logging.getLogger(__name__)


class LocalProvider(STTProvider):
    name = "local"

    _model = None
    _model_lock = threading.Lock()

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self._model_size = model_size
        self._device = device
        self._faster_whisper_available: Optional[bool] = None

    def _check_import(self) -> bool:
        if self._faster_whisper_available is None:
            try:
                import faster_whisper  # noqa: F401
                self._faster_whisper_available = True
            except ImportError:
                self._faster_whisper_available = False
        return self._faster_whisper_available

    def is_available(self) -> bool:
        return self._check_import()

    def _get_model(self):
        """Return singleton model, loading on first call (thread-safe)."""
        if LocalProvider._model is not None:
            return LocalProvider._model

        with LocalProvider._model_lock:
            # Double-checked locking
            if LocalProvider._model is not None:
                return LocalProvider._model

            if not self._check_import():
                raise PermanentError("faster-whisper not installed")

            try:
                from faster_whisper import WhisperModel
                logger.info(
                    "Loading faster-whisper model '%s' on device '%s'",
                    self._model_size,
                    self._device,
                )
                LocalProvider._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type="int8",
                )
                logger.info("faster-whisper model loaded")
            except Exception as e:
                raise TransientError(f"Failed to load faster-whisper model: {e}") from e

        return LocalProvider._model

    def transcribe(self, audio_path: str, language: str = "ru", prompt: str = "") -> TranscriptionResult:
        start = time.monotonic()

        try:
            model = self._get_model()

            kwargs = {
                "language": language,
                "vad_filter": True,
            }
            if prompt:
                kwargs["initial_prompt"] = prompt

            segments, _info = model.transcribe(audio_path, **kwargs)
            text = " ".join(seg.text for seg in segments).strip()

            duration_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(
                text=text,
                provider=self.name,
                duration_ms=duration_ms,
            )

        except PermanentError:
            raise
        except FileNotFoundError as e:
            raise PermanentError(f"Audio file not found: {audio_path}") from e
        except Exception as e:
            raise TransientError(f"Local transcription error: {e}") from e

    @classmethod
    def reset_model(cls) -> None:
        """Force model reload on next transcribe (useful for testing)."""
        with cls._model_lock:
            cls._model = None
