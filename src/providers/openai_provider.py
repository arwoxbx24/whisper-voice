import time
import logging
from typing import Optional

from .base import STTProvider, TranscriptionResult, TransientError, PermanentError

logger = logging.getLogger(__name__)

# Russian tech terms to improve transcription accuracy
_DEFAULT_PROMPT = (
    "Транскрипция на русском языке. "
    "Технические термины: Python, JavaScript, Docker, Kubernetes, API, "
    "GitHub, Linux, Windows, macOS, HTTP, JSON, SQL, MongoDB, Redis, "
    "CI/CD, DevOps, frontend, backend, микросервисы, деплой, коммит."
)


class OpenAIProvider(STTProvider):
    name = "openai"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        if self._api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        except ImportError:
            logger.warning("openai package not installed")
            self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key) and self._client is not None

    def transcribe(self, audio_path: str, language: str = "ru", prompt: str = "") -> TranscriptionResult:
        if not self.is_available():
            raise PermanentError("OpenAI API key not configured or client unavailable")

        effective_prompt = prompt or _DEFAULT_PROMPT
        start = time.monotonic()

        try:
            import openai
            with open(audio_path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language,
                    prompt=effective_prompt,
                )
            duration_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(
                text=response.text.strip(),
                provider=self.name,
                duration_ms=duration_ms,
            )

        except openai.RateLimitError as e:
            raise TransientError(f"OpenAI rate limit: {e}") from e
        except openai.APIConnectionError as e:
            raise TransientError(f"OpenAI connection error: {e}") from e
        except openai.APITimeoutError as e:
            raise TransientError(f"OpenAI timeout: {e}") from e
        except openai.AuthenticationError as e:
            raise PermanentError(f"OpenAI auth failed: {e}") from e
        except openai.PermissionDeniedError as e:
            raise PermanentError(f"OpenAI permission denied: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code in (401, 403):
                raise PermanentError(f"OpenAI HTTP {e.status_code}: {e}") from e
            raise TransientError(f"OpenAI API error {e.status_code}: {e}") from e
        except FileNotFoundError as e:
            raise PermanentError(f"Audio file not found: {audio_path}") from e
        except Exception as e:
            raise TransientError(f"OpenAI unexpected error: {e}") from e
