import abc
from dataclasses import dataclass
from typing import Optional


class TransientError(Exception):
    """Retry-able error: network, rate limit, timeout."""
    pass


class PermanentError(Exception):
    """Do not retry: auth fail, quota exhausted, bad file format."""
    pass


@dataclass
class TranscriptionResult:
    text: str
    provider: str
    duration_ms: float
    confidence: Optional[float] = None


class STTProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: str = "ru", prompt: str = "") -> TranscriptionResult:
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        ...

    def health_check(self) -> bool:
        return self.is_available()
