"""
Phase 3 tests: CircuitBreaker + TranscriptionEngine + provider availability.

Run: cd /root/claude-projects/whisper-voice && python -m pytest tests/test_phase3.py -v
"""
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is importable
sys.path.insert(0, "/root/claude-projects/whisper-voice")

from src.circuit_breaker import CircuitBreaker, CBState, CircuitBreakerOpen
from src.providers.base import TranscriptionResult, TransientError, PermanentError
from src.transcription_engine import TranscriptionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(provider: str = "mock") -> TranscriptionResult:
    return TranscriptionResult(text="hello", provider=provider, duration_ms=10.0)


def _make_mock_provider(name: str, available: bool = True, side_effect=None, result=None):
    """Return a mock STTProvider-like object."""
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = available
    if side_effect is not None:
        provider.transcribe.side_effect = side_effect
    else:
        provider.transcribe.return_value = result or _make_result(name)
    return provider


# ===========================================================================
# CircuitBreaker tests (1-7)
# ===========================================================================


class TestCircuitBreakerInitialState(unittest.TestCase):
    def test_initial_state_closed(self):
        """CircuitBreaker starts in CLOSED state."""
        cb = CircuitBreaker()
        self.assertEqual(cb.state, CBState.CLOSED)


class TestCircuitBreakerOpenOnThreshold(unittest.TestCase):
    def test_open_on_threshold(self):
        """3 consecutive failures → state OPEN."""
        cb = CircuitBreaker(failure_threshold=3)

        def failing():
            raise TransientError("boom")

        for _ in range(3):
            with self.assertRaises(TransientError):
                cb.call(failing)

        self.assertEqual(cb.state, CBState.OPEN)


class TestCircuitBreakerOpenRaises(unittest.TestCase):
    def test_circuit_breaker_open_raises(self):
        """In OPEN state, call() raises CircuitBreakerOpen."""
        cb = CircuitBreaker(failure_threshold=1)

        def failing():
            raise TransientError("boom")

        with self.assertRaises(TransientError):
            cb.call(failing)

        self.assertEqual(cb.state, CBState.OPEN)

        with self.assertRaises(CircuitBreakerOpen):
            cb.call(lambda: "should not run")


class TestCircuitBreakerHalfOpenAfterDuration(unittest.TestCase):
    def test_half_open_after_duration(self):
        """Mock time: advance past open_duration → state HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, open_duration=30.0)

        def failing():
            raise TransientError("boom")

        with self.assertRaises(TransientError):
            cb.call(failing)

        self.assertEqual(cb.state, CBState.OPEN)

        # Simulate time passing beyond open_duration
        cb._opened_at = time.monotonic() - 31.0  # 31s ago, past 30s threshold

        self.assertEqual(cb.state, CBState.HALF_OPEN)


class TestCircuitBreakerCloseOnSuccessInHalfOpen(unittest.TestCase):
    def test_close_on_success_in_half_open(self):
        """Success in HALF_OPEN → state CLOSED."""
        cb = CircuitBreaker(failure_threshold=1, open_duration=30.0)

        def failing():
            raise TransientError("boom")

        with self.assertRaises(TransientError):
            cb.call(failing)

        # Force to HALF_OPEN by backdating _opened_at
        cb._opened_at = time.monotonic() - 31.0

        self.assertEqual(cb.state, CBState.HALF_OPEN)

        # Successful call in HALF_OPEN
        result = cb.call(lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(cb.state, CBState.CLOSED)


class TestCircuitBreakerReopenOnFailureInHalfOpen(unittest.TestCase):
    def test_reopen_on_failure_in_half_open(self):
        """Failure in HALF_OPEN → state OPEN again."""
        cb = CircuitBreaker(failure_threshold=1, open_duration=30.0)

        def failing():
            raise TransientError("boom")

        with self.assertRaises(TransientError):
            cb.call(failing)

        # Force to HALF_OPEN
        cb._opened_at = time.monotonic() - 31.0
        self.assertEqual(cb.state, CBState.HALF_OPEN)

        # Another failure in HALF_OPEN
        with self.assertRaises(TransientError):
            cb.call(failing)

        self.assertEqual(cb.state, CBState.OPEN)


class TestCircuitBreakerSuccessResetsCount(unittest.TestCase):
    def test_success_resets_failure_count(self):
        """2 failures + 1 success + 2 failures → still CLOSED (threshold=3)."""
        cb = CircuitBreaker(failure_threshold=3)

        def failing():
            raise TransientError("boom")

        # 2 failures
        for _ in range(2):
            with self.assertRaises(TransientError):
                cb.call(failing)

        # 1 success — resets failure count
        cb.call(lambda: "ok")

        # 2 more failures (total would be 4, but count was reset to 0 after success)
        for _ in range(2):
            with self.assertRaises(TransientError):
                cb.call(failing)

        # Should still be CLOSED (only 2 consecutive failures since last reset)
        self.assertEqual(cb.state, CBState.CLOSED)


# ===========================================================================
# TranscriptionEngine tests (8-12)
# ===========================================================================


class TestTranscriptionEngineFirstProviderSucceeds(unittest.TestCase):
    def test_first_provider_succeeds(self):
        """Mock provider succeeds → result returned."""
        expected = _make_result("p1")
        p1 = _make_mock_provider("p1", result=expected)
        p2 = _make_mock_provider("p2")

        engine = TranscriptionEngine([p1, p2])
        result = engine.transcribe("audio.wav")

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.provider, "p1")
        p2.transcribe.assert_not_called()


class TestTranscriptionEngineFailoverOnTransientError(unittest.TestCase):
    def test_failover_on_transient_error(self):
        """First provider raises TransientError → second called → succeeds."""
        p1 = _make_mock_provider("p1", side_effect=TransientError("network"))
        p2 = _make_mock_provider("p2", result=_make_result("p2"))

        engine = TranscriptionEngine([p1, p2])
        result = engine.transcribe("audio.wav")

        self.assertEqual(result.provider, "p2")
        p1.transcribe.assert_called_once()
        p2.transcribe.assert_called_once()


class TestTranscriptionEnginePermanentErrorNotRetried(unittest.TestCase):
    def test_permanent_error_not_retried(self):
        """PermanentError → raised immediately, second provider NOT called."""
        p1 = _make_mock_provider("p1", side_effect=PermanentError("auth fail"))
        p2 = _make_mock_provider("p2")

        engine = TranscriptionEngine([p1, p2])

        with self.assertRaises(PermanentError):
            engine.transcribe("audio.wav")

        p2.transcribe.assert_not_called()


class TestTranscriptionEngineAllProvidersFail(unittest.TestCase):
    def test_all_providers_fail(self):
        """All providers raise TransientError → TransientError raised."""
        p1 = _make_mock_provider("p1", side_effect=TransientError("net1"))
        p2 = _make_mock_provider("p2", side_effect=TransientError("net2"))

        engine = TranscriptionEngine([p1, p2])

        with self.assertRaises(TransientError):
            engine.transcribe("audio.wav")


class TestTranscriptionEngineCircuitOpensForFailingProvider(unittest.TestCase):
    def test_circuit_opens_for_failing_provider(self):
        """3 failures → 4th call skips first provider (circuit OPEN), uses second."""
        p1 = _make_mock_provider("p1", side_effect=TransientError("net"))
        p2 = _make_mock_provider("p2", result=_make_result("p2"))

        engine = TranscriptionEngine([p1, p2], failure_threshold=3)

        # 3 calls where p1 fails → p2 handles each (p1 circuit trips after 3rd failure)
        for _ in range(3):
            result = engine.transcribe("audio.wav")
            self.assertEqual(result.provider, "p2")

        # Circuit for p1 should now be OPEN
        breaker = engine.get_breaker("p1")
        self.assertEqual(breaker.state, CBState.OPEN)

        # 4th call: p1 circuit is OPEN → skipped → p2 called directly
        p1.transcribe.reset_mock()
        p2.transcribe.reset_mock()
        result = engine.transcribe("audio.wav")
        self.assertEqual(result.provider, "p2")
        p1.transcribe.assert_not_called()
        p2.transcribe.assert_called_once()


# ===========================================================================
# Provider availability tests (13-14)
# ===========================================================================


class TestOpenAIProviderAvailability(unittest.TestCase):
    def test_openai_provider_is_available_with_key(self):
        """With api_key and openai installed → is_available() True."""
        from src.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        with patch("src.providers.openai_provider.OpenAIProvider._init_client") as mock_init:
            provider = OpenAIProvider.__new__(OpenAIProvider)
            setattr(provider, "_api_key", "test-placeholder")  # noqa: test fixture only
            provider._client = mock_client  # simulate successful init

            self.assertTrue(provider.is_available())

    def test_openai_provider_not_available_without_key(self):
        """Without api_key → is_available() False."""
        from src.providers.openai_provider import OpenAIProvider

        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY from environment if present
            import os
            os.environ.pop("OPENAI_API_KEY", None)

            provider = OpenAIProvider(api_key=None)
            # api_key will be "" (env not set), client = None
            self.assertFalse(provider.is_available())


class TestLocalProviderNotAvailableWithoutModule(unittest.TestCase):
    def test_local_provider_not_available_without_module(self):
        """Mock import failure → is_available() returns False."""
        from src.providers.local_provider import LocalProvider

        provider = LocalProvider.__new__(LocalProvider)
        provider._faster_whisper_available = None  # force lazy check

        with patch.dict("sys.modules", {"faster_whisper": None}):
            # Remove cached result to force re-check
            provider._faster_whisper_available = None
            result = provider._check_import()
            self.assertFalse(result)
            self.assertFalse(provider.is_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
