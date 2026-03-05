"""
Thread-safe state machine for the whisper-voice recording lifecycle.

States: IDLE -> RECORDING -> PROCESSING -> INSERTING -> IDLE
Cancel/error paths: RECORDING -> IDLE, PROCESSING -> IDLE
"""

import logging
import threading
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    INSERTING = "inserting"


VALID_TRANSITIONS: Dict[State, List[State]] = {
    State.IDLE: [State.RECORDING],
    State.RECORDING: [State.PROCESSING, State.IDLE],
    State.PROCESSING: [State.INSERTING, State.IDLE],
    State.INSERTING: [State.IDLE],
}


class StateMachine:
    """Thread-safe state machine for recording lifecycle management."""

    def __init__(self) -> None:
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._on_enter_callbacks: Dict[State, List[Callable[[], None]]] = {
            s: [] for s in State
        }
        self._on_exit_callbacks: Dict[State, List[Callable[[], None]]] = {
            s: [] for s in State
        }

    @property
    def current(self) -> State:
        """Return current state (thread-safe read)."""
        with self._lock:
            return self._state

    def on_enter(self, state: State, callback: Callable[[], None]) -> None:
        """Register a callback to fire when entering the given state."""
        with self._lock:
            self._on_enter_callbacks[state].append(callback)

    def on_exit(self, state: State, callback: Callable[[], None]) -> None:
        """Register a callback to fire when exiting the given state."""
        with self._lock:
            self._on_exit_callbacks[state].append(callback)

    def transition(self, new_state: State) -> bool:
        """
        Attempt to transition to new_state.

        Validates against VALID_TRANSITIONS, then fires on_exit/on_enter
        callbacks OUTSIDE the lock to prevent deadlock.

        Returns True if transition succeeded, False if invalid.
        """
        with self._lock:
            old_state = self._state
            allowed = VALID_TRANSITIONS.get(old_state, [])
            if new_state not in allowed:
                logger.warning(
                    "Invalid transition %s -> %s (allowed: %s)",
                    old_state.value,
                    new_state.value,
                    [s.value for s in allowed],
                )
                return False
            self._state = new_state
            exit_callbacks = list(self._on_exit_callbacks[old_state])
            enter_callbacks = list(self._on_enter_callbacks[new_state])

        # Fire callbacks outside the lock to prevent deadlock
        logger.info("State transition: %s -> %s", old_state.value, new_state.value)
        for cb in exit_callbacks:
            try:
                cb()
            except Exception:
                logger.exception(
                    "on_exit callback error during transition %s -> %s",
                    old_state.value,
                    new_state.value,
                )
        for cb in enter_callbacks:
            try:
                cb()
            except Exception:
                logger.exception(
                    "on_enter callback error during transition %s -> %s",
                    old_state.value,
                    new_state.value,
                )
        return True

    def reset(self) -> None:
        """
        Force state to IDLE for crash recovery.

        Bypasses transition validation. Fires on_enter IDLE callbacks.
        """
        with self._lock:
            old_state = self._state
            self._state = State.IDLE
            enter_callbacks = list(self._on_enter_callbacks[State.IDLE])

        logger.warning(
            "State machine reset: %s -> IDLE (crash recovery)", old_state.value
        )
        for cb in enter_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("on_enter callback error during reset to IDLE")
