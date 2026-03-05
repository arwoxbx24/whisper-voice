"""
Lightweight background network monitor.

Polls a set of well-known TCP endpoints at a fixed interval and fires
callbacks when connectivity state changes (connected ↔ disconnected).
"""

import socket
import threading
import time
from typing import Callable, List, Optional, Tuple

# Endpoints tried in order; at least one reachable = connected
_PROBE_ENDPOINTS: List[Tuple[str, int]] = [
    ("api.openai.com", 443),
    ("8.8.8.8", 53),
    ("1.1.1.1", 53),
]

_CHECK_INTERVAL: float = 10.0   # seconds between polls
_PROBE_TIMEOUT: float = 3.0     # seconds per TCP probe attempt


def _probe_endpoint(host: str, port: int, timeout: float) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_connectivity(
    endpoints: List[Tuple[str, int]] = _PROBE_ENDPOINTS,
    timeout: float = _PROBE_TIMEOUT,
) -> bool:
    """Return True if at least one endpoint is reachable."""
    for host, port in endpoints:
        if _probe_endpoint(host, port, timeout):
            return True
    return False


class NetworkMonitor:
    """
    Daemon thread that monitors internet connectivity and fires callbacks
    on state changes.

    Usage::

        def on_up():
            print("Network restored — flushing queue")

        def on_down():
            print("Network lost — queuing audio locally")

        monitor = NetworkMonitor(on_connected=on_up, on_disconnected=on_down)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(
        self,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
        check_interval: float = _CHECK_INTERVAL,
        probe_timeout: float = _PROBE_TIMEOUT,
        endpoints: Optional[List[Tuple[str, int]]] = None,
    ):
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._check_interval = check_interval
        self._probe_timeout = probe_timeout
        self._endpoints = endpoints or list(_PROBE_ENDPOINTS)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Optimistic initial state — will be confirmed on first poll
        self._connected: bool = True
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread (daemon=True)."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="NetworkMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval + self._probe_timeout + 1)
            self._thread = None

    @property
    def is_connected(self) -> bool:
        """Thread-safe read of the current connectivity state."""
        with self._state_lock:
            return self._connected

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Poll loop — runs in daemon thread."""
        while not self._stop_event.is_set():
            now_connected = _check_connectivity(
                endpoints=self._endpoints,
                timeout=self._probe_timeout,
            )
            self._handle_state_change(now_connected)

            # Interruptible sleep: wakes immediately on stop()
            self._stop_event.wait(timeout=self._check_interval)

    def _handle_state_change(self, now_connected: bool) -> None:
        """Update state and fire callback only when state actually changes."""
        with self._state_lock:
            previous = self._connected
            self._connected = now_connected

        if now_connected and not previous:
            # Transition: disconnected → connected
            if self._on_connected is not None:
                try:
                    self._on_connected()
                except Exception:  # noqa: BLE001 — callbacks must not crash monitor
                    pass

        elif not now_connected and previous:
            # Transition: connected → disconnected
            if self._on_disconnected is not None:
                try:
                    self._on_disconnected()
                except Exception:  # noqa: BLE001
                    pass
