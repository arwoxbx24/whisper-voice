import time
import logging
from typing import Optional

from .base import STTProvider, TranscriptionResult, TransientError, PermanentError

logger = logging.getLogger(__name__)


class DeepgramProvider(STTProvider):
    name = "deepgram"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self._client = None
        if self._api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            from deepgram import DeepgramClient
            self._client = DeepgramClient(self._api_key)
        except ImportError:
            logger.warning("deepgram-sdk package not installed")
            self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key) and self._client is not None

    def transcribe(self, audio_path: str, language: str = "ru", prompt: str = "") -> TranscriptionResult:
        if not self.is_available():
            raise PermanentError("Deepgram API key not configured or client unavailable")

        start = time.monotonic()

        try:
            from deepgram import PrerecordedOptions

            with open(audio_path, "rb") as f:
                data = f.read()

            options = PrerecordedOptions(
                model="nova-3",
                language=language,
                smart_format=True,
                punctuate=True,
            )

            response = self._client.listen.rest.v("1").transcribe_file(
                {"buffer": data, "mimetype": "audio/wav"},
                options,
            )

            duration_ms = (time.monotonic() - start) * 1000

            # Extract transcript from response
            try:
                text = (
                    response.results.channels[0]
                    .alternatives[0]
                    .transcript
                )
                confidence = (
                    response.results.channels[0]
                    .alternatives[0]
                    .confidence
                )
            except (AttributeError, IndexError) as e:
                raise TransientError(f"Deepgram unexpected response structure: {e}") from e

            return TranscriptionResult(
                text=text.strip(),
                provider=self.name,
                duration_ms=duration_ms,
                confidence=float(confidence) if confidence is not None else None,
            )

        except PermanentError:
            raise
        except FileNotFoundError as e:
            raise PermanentError(f"Audio file not found: {audio_path}") from e
        except Exception as e:
            msg = str(e).lower()
            if any(kw in msg for kw in ("unauthorized", "forbidden", "invalid api key", "401", "403")):
                raise PermanentError(f"Deepgram auth error: {e}") from e
            raise TransientError(f"Deepgram error: {e}") from e
