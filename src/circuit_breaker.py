"""
Circuit Breaker for STT provider fault isolation.

States: CLOSED → OPEN → HALF_OPEN → CLOSED

- CLOSED: normal operation, failures counted
- OPEN: fast-fail all calls, waits open_duration before transitioning
- HALF_OPEN: one test call allowed; success → CLOSED, failure → OPEN
"""
import threading
import time
from enum import Enum
from typing import Callable, Any


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when circuit is OPEN and call is blocked."""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Args:
        failure_threshold: consecutive failures to trip OPEN (default 3)
        open_duration: seconds to wait in OPEN before HALF_OPEN (default 60)
    """

    def __init__(self, failure_threshold: int = 3, open_duration: float = 60.0):
        self.failure_threshold = failure_threshold
        self.open_duration = open_duration

        self._state = CBState.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CBState:
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CBState:
        """Must be called under self._lock."""
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.open_duration:
                self._state = CBState.HALF_OPEN
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute func through the circuit breaker.

        Raises:
            CircuitBreakerOpen: if state is OPEN
            Exception: whatever func raises (and records as failure)
        """
        with self._lock:
            state = self._get_state()
            if state == CBState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit is OPEN, failing fast (opened {time.monotonic() - self._opened_at:.1f}s ago)"
                )

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self):
        with self._lock:
            self._failure_count += 1
            state = self._get_state()
            if state == CBState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                self._failure_count = 0

    def _record_success(self):
        with self._lock:
            state = self._get_state()
            if state == CBState.HALF_OPEN:
                self._state = CBState.CLOSED
                self._failure_count = 0
            elif state == CBState.CLOSED:
                self._failure_count = 0

    def reset(self):
        """Force reset to CLOSED state."""
        with self._lock:
            self._state = CBState.CLOSED
            self._failure_count = 0
            self._opened_at = 0.0
