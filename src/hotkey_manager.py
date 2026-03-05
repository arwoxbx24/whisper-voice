"""
hotkey_manager.py — Global hotkey listener for Whisper Voice.

Supports keyboard shortcuts and mouse button triggers.
Works on Windows (primary), Linux, macOS via pynput.

Modes:
  toggle — press to start recording, press again to stop
  hold   — hold key to record, release to stop

Usage:
    manager = HotkeyManager(hotkey="<ctrl>+<shift>+space", mode="toggle")
    manager.set_callback(on_activate=start_recording, on_deactivate=stop_recording)
    manager.start()
    # ... later ...
    manager.stop()

Mouse trigger:
    manager = HotkeyManager(mouse_button="middle", mode="hold")
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Set

from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button

logger = logging.getLogger(__name__)


def _get_key_safe(name: str):
    """Safely get a Key enum member by name."""
    try:
        return Key[name]
    except KeyError:
        return None


def _build_mouse_button_map() -> dict:
    """Build map of mouse button aliases to pynput Button members."""
    from pynput.mouse import Button
    m = {}
    for alias, attr in [("x1", "x1"), ("x2", "x2"), ("button4", "x1"), ("button5", "x2"),
                         ("middle", "middle"), ("left", "left"), ("right", "right")]:
        if hasattr(Button, attr):
            m[alias] = getattr(Button, attr)
    return m


# Public module-level map of mouse button aliases to pynput Button constants
MOUSE_BUTTON_MAP = _build_mouse_button_map()

# Keep private alias for internal use (backward compat)
_MOUSE_BUTTON_MAP = MOUSE_BUTTON_MAP


KNOWN_CONFLICTS = {
    "<ctrl>+<alt>+<delete>": "System shortcut (task manager/reboot)",
    "<cmd>+space": "macOS Spotlight",
    "<ctrl>+<alt>+t": "Ubuntu terminal shortcut",
    "<super>+l": "Lock screen",
    "<ctrl>+<alt>+l": "Lock screen (Linux)",
}

SUGGESTED_ALTERNATIVES = [
    "<ctrl>+<shift>+space", "<ctrl>+<alt>+space", "<ctrl>+<shift>+r",
    "<ctrl>+<alt>+r", "<ctrl>+<shift>+f12", "<alt>+f9",
]


def check_hotkey_available(hotkey_str: str) -> tuple:
    """Check if hotkey conflicts with known system shortcuts.
    Returns (is_available: bool, reason: str)."""
    normalized = hotkey_str.lower().replace(" ", "")
    for conflict, reason in KNOWN_CONFLICTS.items():
        if normalized == conflict.lower().replace(" ", ""):
            return (False, reason)
    return (True, "")


def _parse_hotkey(hotkey_str: str) -> frozenset:
    """
    Parse a hotkey string like "<ctrl>+<shift>+space" into a frozenset
    of canonical pynput key representations (used for matching pressed keys).

    Supported formats:
      "<ctrl>+<shift>+space"
      "<cmd>+<alt>+r"
      "a"  (single key)
    """
    parts = [p.strip() for p in hotkey_str.lower().split("+")]
    canonical: set = set()
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            name = part[1:-1]
            # Map common aliases
            aliases = {
                "ctrl": "ctrl_l",
                "alt": "alt_l",
                "shift": "shift_l",
                "cmd": "cmd",
                "win": "cmd",
                "super": "cmd",
                "meta": "cmd",
            }
            key_name = aliases.get(name, name)
            try:
                canonical.add(Key[key_name])
            except KeyError:
                # Try without _l suffix fallback
                try:
                    canonical.add(Key[name])
                except KeyError:
                    logger.warning("Unknown key name in hotkey: %s", part)
        else:
            # Regular character key
            if len(part) == 1:
                canonical.add(KeyCode.from_char(part))
            else:
                # Named key without angle brackets e.g. "space", "enter"
                try:
                    canonical.add(Key[part])
                except KeyError:
                    logger.warning("Unknown key name in hotkey: %s", part)
    return frozenset(canonical)


def _keys_match(pressed: Set, hotkey: frozenset) -> bool:
    """
    Check whether all keys in hotkey are currently pressed.
    Handles left/right modifier variants (ctrl_l / ctrl_r both satisfy <ctrl>).
    """
    for required in hotkey:
        if isinstance(required, Key):
            # Check for left or right variant
            name = required.name
            if name.endswith("_l"):
                base = name[:-2]
                right_variant = _get_key_safe(base + "_r") if hasattr(Key, base + "_r") else None
                left_key = required
                # Satisfied if left or right variant is pressed
                found = left_key in pressed
                if not found and right_variant is not None:
                    found = right_variant in pressed
                if not found:
                    # Also check the bare modifier (some platforms report it)
                    bare = _get_key_safe(base) if hasattr(Key, base) else None
                    found = bare in pressed if bare else False
                if not found:
                    return False
            else:
                if required not in pressed:
                    return False
        else:
            if required not in pressed:
                return False
    return True


class HotkeyManager:
    """
    Global hotkey / mouse button listener.

    Parameters
    ----------
    hotkey : str or None
        Keyboard shortcut string, e.g. "<ctrl>+<shift>+space".
        Set to None to disable keyboard trigger.
    mouse_button : str or None
        Mouse button name: "middle", "left", "right", "button4", "button5".
        Set to None to disable mouse trigger.
    mode : str
        "toggle" — first activation starts, second stops.
        "hold"   — held activation records, release stops.
    """

    def __init__(
        self,
        hotkey: Optional[str] = "<ctrl>+<shift>+space",
        mouse_button: Optional[str] = None,
        mode: str = "toggle",
    ) -> None:
        if mode not in ("toggle", "hold"):
            raise ValueError(f"mode must be 'toggle' or 'hold', got {mode!r}")

        self._hotkey_str = hotkey
        self._hotkey: Optional[frozenset] = _parse_hotkey(hotkey) if hotkey else None
        self._mouse_button_str = mouse_button
        self._mouse_button: Optional[Button] = (
            _MOUSE_BUTTON_MAP.get(mouse_button.lower()) if mouse_button else None
        )
        self._mode = mode

        self._on_activate: Optional[Callable[[], None]] = None
        self._on_deactivate: Optional[Callable[[], None]] = None

        self._pressed_keys: Set = set()
        self._is_active = False          # recording in progress
        self._hotkey_triggered = False   # True while hotkey combo is held (for hold mode dedup)
        self._lock = threading.Lock()

        self._kb_listener: Optional[keyboard.Listener] = None
        self._mouse_listener: Optional[mouse.Listener] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callback(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        """Set callbacks invoked when recording starts / stops."""
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def start(self) -> None:
        """Start listening for hotkeys in the background (non-blocking)."""
        if self._kb_listener is not None or self._mouse_listener is not None:
            logger.warning("HotkeyManager.start() called while already running — ignored")
            return

        if self._hotkey is not None:
            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._kb_listener.start()
            logger.debug("Keyboard listener started (hotkey=%s, mode=%s)", self._hotkey_str, self._mode)

        if self._mouse_button is not None:
            self._mouse_listener = mouse.Listener(
                on_click=self._on_mouse_click,
            )
            self._mouse_listener.start()
            logger.debug("Mouse listener started (button=%s, mode=%s)", self._mouse_button_str, self._mode)

    def stop(self) -> None:
        """Stop all listeners and reset state."""
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._kb_listener = None

        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

        with self._lock:
            if self._is_active:
                self._fire_deactivate()
            self._pressed_keys.clear()
            self._hotkey_triggered = False

        logger.debug("HotkeyManager stopped")

    def update_hotkey(self, new_hotkey: str) -> None:
        """
        Change the keyboard hotkey combination at runtime.
        Stops and restarts the keyboard listener if active.
        """
        was_running = self._kb_listener is not None

        if was_running:
            self._kb_listener.stop()
            self._kb_listener = None

        with self._lock:
            self._hotkey_str = new_hotkey
            self._hotkey = _parse_hotkey(new_hotkey)
            self._pressed_keys.clear()
            self._hotkey_triggered = False
            if self._is_active:
                self._fire_deactivate()

        if was_running:
            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._kb_listener.start()
            logger.debug("Hotkey updated to %s", new_hotkey)

    def update_mouse_button(self, new_button: Optional[str]) -> None:
        """Change mouse button trigger at runtime."""
        was_running = self._mouse_listener is not None

        if was_running:
            self._mouse_listener.stop()
            self._mouse_listener = None

        with self._lock:
            self._mouse_button_str = new_button
            self._mouse_button = (
                _MOUSE_BUTTON_MAP.get(new_button.lower()) if new_button else None
            )

        if was_running and self._mouse_button is not None:
            self._mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
            self._mouse_listener.start()
            logger.debug("Mouse button updated to %s", new_button)

    @property
    def is_active(self) -> bool:
        """True if recording is currently in progress."""
        return self._is_active

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("toggle", "hold"):
            raise ValueError(f"mode must be 'toggle' or 'hold', got {value!r}")
        with self._lock:
            self._mode = value
            if self._is_active:
                self._fire_deactivate()
            self._hotkey_triggered = False

    # ------------------------------------------------------------------
    # Internal: keyboard events
    # ------------------------------------------------------------------

    def _canonical(self, key) -> object:
        """Return canonical form of a key for set membership."""
        if isinstance(key, KeyCode) and key.char is not None:
            return KeyCode.from_char(key.char.lower())
        return key

    def _on_key_press(self, key) -> None:
        canon = self._canonical(key)
        with self._lock:
            self._pressed_keys.add(canon)
            self._check_keyboard_trigger(pressed=True)

    def _on_key_release(self, key) -> None:
        canon = self._canonical(key)
        with self._lock:
            self._check_keyboard_trigger(pressed=False)
            self._pressed_keys.discard(canon)

            if self._mode == "hold" and self._hotkey_triggered:
                # Any key in the combo released → end hold
                if not _keys_match(self._pressed_keys, self._hotkey):
                    self._hotkey_triggered = False
                    if self._is_active:
                        self._fire_deactivate()
            elif self._mode == "toggle":
                # Reset trigger flag when hotkey keys are released so next press fires again
                if not _keys_match(self._pressed_keys, self._hotkey):
                    self._hotkey_triggered = False

    def _check_keyboard_trigger(self, *, pressed: bool) -> None:
        """Called with lock held. Evaluate combo match and fire callbacks."""
        if self._hotkey is None:
            return

        combo_active = _keys_match(self._pressed_keys, self._hotkey)

        if self._mode == "toggle":
            if combo_active and pressed and not self._hotkey_triggered:
                self._hotkey_triggered = True
                self._toggle()
        elif self._mode == "hold":
            if combo_active and not self._hotkey_triggered:
                self._hotkey_triggered = True
                if not self._is_active:
                    self._fire_activate()

    # ------------------------------------------------------------------
    # Internal: mouse events
    # ------------------------------------------------------------------

    def _on_mouse_click(self, x: int, y: int, button: Button, is_press: bool) -> None:
        if button != self._mouse_button:
            return

        with self._lock:
            if self._mode == "toggle":
                if is_press:
                    self._toggle()
            elif self._mode == "hold":
                if is_press and not self._is_active:
                    self._fire_activate()
                elif not is_press and self._is_active:
                    self._fire_deactivate()

    # ------------------------------------------------------------------
    # Internal: state helpers (called with lock held)
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        if self._is_active:
            self._fire_deactivate()
        else:
            self._fire_activate()

    def _fire_activate(self) -> None:
        """Mark active and call on_activate callback (outside lock for safety)."""
        self._is_active = True
        cb = self._on_activate
        if cb is not None:
            threading.Thread(target=cb, daemon=True, name="hotkey-activate").start()

    def _fire_deactivate(self) -> None:
        """Mark inactive and call on_deactivate callback (outside lock for safety)."""
        self._is_active = False
        cb = self._on_deactivate
        if cb is not None:
            threading.Thread(target=cb, daemon=True, name="hotkey-deactivate").start()


# ---------------------------------------------------------------------------
# Demo / self-test (run directly: python hotkey_manager.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    print("HotkeyManager demo")
    print("  Keyboard: Ctrl+Shift+Space  (toggle mode)")
    print("  Mouse:    middle click      (toggle mode)")
    print("  Press Ctrl+C to exit\n")

    def on_start():
        print("[RECORDING STARTED]")

    def on_stop():
        print("[RECORDING STOPPED]")

    mgr = HotkeyManager(
        hotkey="<ctrl>+<shift>+space",
        mouse_button="middle",
        mode="toggle",
    )
    mgr.set_callback(on_activate=on_start, on_deactivate=on_stop)
    mgr.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        print("Done.")
