"""
TranscriptionEngine: multi-provider STT with circuit breaker and failover.

Tries providers in order. On TransientError → try next provider.
On PermanentError → stop immediately (do not retry).
Each provider has its own CircuitBreaker; open breakers are skipped.
"""
import logging
from typing import List, Optional

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from .providers.base import STTProvider, TranscriptionResult, TransientError, PermanentError

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """
    Orchestrates multiple STT providers with circuit breakers.

    Args:
        providers: ordered list of providers (first = preferred)
        failure_threshold: failures before circuit opens per provider (default 3)
        open_duration: seconds before circuit attempts recovery (default 60)
    """

    def __init__(
        self,
        providers: List[STTProvider],
        failure_threshold: int = 3,
        open_duration: float = 60.0,
    ):
        self._providers = providers
        self._breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(failure_threshold=failure_threshold, open_duration=open_duration)
            for p in providers
        }

    def transcribe(self, audio_path: str, language: str = "ru", prompt: str = "") -> TranscriptionResult:
        """
        Transcribe audio using the first available, non-open provider.

        Raises:
            TransientError: if all providers fail with transient errors
            PermanentError: if any provider raises a permanent error
        """
        last_transient: Optional[TransientError] = None

        for provider in self._providers:
            if not provider.is_available():
                logger.debug("Provider %s not available, skipping", provider.name)
                continue

            breaker = self._breakers[provider.name]

            try:
                result = breaker.call(provider.transcribe, audio_path, language, prompt)
                return result
            except CircuitBreakerOpen:
                logger.warning("Circuit OPEN for provider %s, skipping", provider.name)
                continue
            except PermanentError:
                raise
            except TransientError as e:
                last_transient = e
                logger.warning("Provider %s transient error: %s, trying next", provider.name, e)
                continue

        if last_transient is not None:
            raise last_transient
        raise TransientError("No available providers")

    def get_breaker(self, provider_name: str) -> Optional[CircuitBreaker]:
        """Return the circuit breaker for a named provider."""
        return self._breakers.get(provider_name)

    def provider_status(self) -> dict:
        """Return {provider_name: circuit_state} for diagnostics."""
        return {
            name: breaker.state.value
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()
