"""
Phase 1 tests: StateMachine + HotkeyManager

Covers:
  - State machine lifecycle, callbacks, thread-safety
  - Hotkey toggle/hold modes, conflict detection helpers, key utilities

All tests run without real keyboard/mouse hardware (pynput patched at import).
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub pynput BEFORE importing any project code that imports pynput.
# This avoids the X11 "Bad display name" ImportError on headless CI/servers.
# ---------------------------------------------------------------------------

# Create a minimal stub for pynput.keyboard
class _FakeKey:
    """Enum-like stub for pynput.keyboard.Key."""

    def __init__(self, name):
        self.name = name
        self._value_ = name

    def __repr__(self):
        return f"Key.{self.name}"

    def __hash__(self):
        return hash(("Key", self.name))

    def __eq__(self, other):
        return isinstance(other, _FakeKey) and self.name == other.name


def _make_fake_key_cls():
    """Build a Key-like namespace with common modifier attributes."""
    names = [
        "ctrl_l", "ctrl_r", "alt_l", "alt_r", "shift_l", "shift_r",
        "cmd", "delete", "space", "enter", "backspace", "tab",
    ]
    cls_dict = {"_members": {}}
    for n in names:
        key = _FakeKey(n)
        cls_dict[n] = key
        cls_dict["_members"][n] = key

    def class_getitem(name):
        """Mimic Key[name] — raises KeyError if not found."""
        members = cls_dict["_members"]
        if name in members:
            return members[name]
        raise KeyError(name)

    cls_dict["__class_getitem__"] = class_getitem
    cls_dict["__getitem__"] = staticmethod(class_getitem)
    # Allow `Key[name]` syntax via __class_getitem__ at the *instance* level
    FakeKeyCls = type("Key", (), cls_dict)
    # Make `Key[name]` work (subscript on the class object)
    FakeKeyCls.__class_getitem__ = classmethod(lambda cls, name: class_getitem(name))

    # Patch __getitem__ on the metaclass so Key[name] works
    meta = type(FakeKeyCls)
    try:
        meta.__getitem__ = lambda self, name: class_getitem(name)
    except TypeError:
        pass  # Can't patch `type` itself

    return FakeKeyCls, class_getitem


_FakeKeyCls, _key_getitem = _make_fake_key_cls()

# Provide Key[name] support by wrapping the class in a dict-like proxy
class _KeyProxy:
    """Acts as both a namespace (Key.ctrl_l) and a subscriptable (Key['ctrl_l'])."""

    def __init__(self, cls, getitem_fn):
        self._cls = cls
        self._getitem = getitem_fn
        # Copy attributes
        for name, val in cls.__dict__.items():
            if not name.startswith("__"):
                setattr(self, name, val)

    def __getitem__(self, name):
        return self._getitem(name)

    def __call__(self, *a, **kw):
        return self._cls(*a, **kw)


_KeyNamespace = _KeyProxy(_FakeKeyCls, _key_getitem)


class _FakeKeyCode:
    """Stub for pynput.keyboard.KeyCode."""

    def __init__(self, char=None, vk=None):
        self.char = char
        self.vk = vk

    @classmethod
    def from_char(cls, char):
        return cls(char=char)

    def __hash__(self):
        return hash(("KeyCode", self.char, self.vk))

    def __eq__(self, other):
        return (
            isinstance(other, _FakeKeyCode)
            and self.char == other.char
            and self.vk == other.vk
        )

    def __repr__(self):
        return f"KeyCode(char={self.char!r})"


class _FakeButton:
    """Stub for pynput.mouse.Button."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Button.{self.name}"

    def __hash__(self):
        return hash(("Button", self.name))

    def __eq__(self, other):
        return isinstance(other, _FakeButton) and self.name == other.name


_FakeButtonLeft = _FakeButton("left")
_FakeButtonRight = _FakeButton("right")
_FakeButtonMiddle = _FakeButton("middle")
_FakeButtonX1 = _FakeButton("x1")
_FakeButtonX2 = _FakeButton("x2")


class _FakeButtonNamespace:
    left = _FakeButtonLeft
    right = _FakeButtonRight
    middle = _FakeButtonMiddle
    x1 = _FakeButtonX1
    x2 = _FakeButtonX2


class _FakeListener:
    """Stub for pynput keyboard/mouse Listener — does nothing."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


# Inject stubs into sys.modules BEFORE any project import
_kb_mod = MagicMock()
_kb_mod.Key = _KeyNamespace
_kb_mod.KeyCode = _FakeKeyCode
_kb_mod.Listener = _FakeListener

_mouse_mod = MagicMock()
_mouse_mod.Button = _FakeButtonNamespace()
_mouse_mod.Listener = _FakeListener

_pynput_mod = MagicMock()
_pynput_mod.keyboard = _kb_mod
_pynput_mod.mouse = _mouse_mod

sys.modules.setdefault("pynput", _pynput_mod)
sys.modules.setdefault("pynput.keyboard", _kb_mod)
sys.modules.setdefault("pynput.mouse", _mouse_mod)

# ---------------------------------------------------------------------------
# Now it is safe to import project modules
# ---------------------------------------------------------------------------

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from state_machine import State, StateMachine  # noqa: E402
from hotkey_manager import (  # noqa: E402
    HotkeyManager,
    _MOUSE_BUTTON_MAP,
    _get_key_safe,
    _parse_hotkey,
    _keys_match,
    Key,
    KeyCode,
    Button,
)


# ---------------------------------------------------------------------------
# StateMachine tests (1–9)
# ---------------------------------------------------------------------------

class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = StateMachine()

    # 1. Initial state is IDLE
    def test_initial_state_is_idle(self):
        self.assertEqual(self.sm.current, State.IDLE)

    # 2. Valid transition IDLE → RECORDING
    def test_valid_transition_idle_to_recording(self):
        result = self.sm.transition(State.RECORDING)
        self.assertTrue(result)
        self.assertEqual(self.sm.current, State.RECORDING)

    # 3. Invalid transition IDLE → PROCESSING
    def test_invalid_transition_idle_to_processing(self):
        result = self.sm.transition(State.PROCESSING)
        self.assertFalse(result)
        self.assertEqual(self.sm.current, State.IDLE)  # state unchanged

    # 4. RECORDING → IDLE (cancel path)
    def test_cancel_recording_to_idle(self):
        self.sm.transition(State.RECORDING)
        result = self.sm.transition(State.IDLE)
        self.assertTrue(result)
        self.assertEqual(self.sm.current, State.IDLE)

    # 5. Full happy-path cycle
    def test_full_cycle(self):
        transitions = [
            State.RECORDING,
            State.PROCESSING,
            State.INSERTING,
            State.IDLE,
        ]
        for target in transitions:
            ok = self.sm.transition(target)
            self.assertTrue(ok, f"Expected transition to {target} to succeed")
        self.assertEqual(self.sm.current, State.IDLE)

    # 6. on_enter callback fires on transition into RECORDING
    def test_on_enter_callback_fires(self):
        cb = MagicMock()
        self.sm.on_enter(State.RECORDING, cb)
        self.sm.transition(State.RECORDING)
        cb.assert_called_once()

    # 7. on_exit callback fires when leaving IDLE
    def test_on_exit_callback_fires(self):
        cb = MagicMock()
        self.sm.on_exit(State.IDLE, cb)
        self.sm.transition(State.RECORDING)
        cb.assert_called_once()

    # 8. reset() from PROCESSING forces state to IDLE
    def test_reset_to_idle(self):
        self.sm.transition(State.RECORDING)
        self.sm.transition(State.PROCESSING)
        self.assertEqual(self.sm.current, State.PROCESSING)
        self.sm.reset()
        self.assertEqual(self.sm.current, State.IDLE)

    # 9. Thread safety: 100 transitions from multiple threads, no crashes
    def test_thread_safety(self):
        errors = []

        def worker():
            try:
                for _ in range(10):
                    # Drive a mini cycle; ignore invalid-transition False returns
                    self.sm.transition(State.RECORDING)
                    self.sm.transition(State.IDLE)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [], f"Thread-safety errors: {errors}")
        # State must be valid after all chaos
        self.assertIn(self.sm.current, list(State))


# ---------------------------------------------------------------------------
# HotkeyManager tests (10–15)
# ---------------------------------------------------------------------------

class TestHotkeyManager(unittest.TestCase):
    """Tests that exercise HotkeyManager internals without real listeners."""

    def _make_manager(self, mode="toggle", hotkey="<ctrl>+<shift>+space"):
        mgr = HotkeyManager(hotkey=hotkey, mouse_button=None, mode=mode)
        return mgr

    # 10. Toggle mode: 5 press/release cycles → 5 activate + 5 deactivate
    def test_toggle_multiple_cycles(self):
        mgr = self._make_manager(mode="toggle")
        activate_count = [0]
        deactivate_count = [0]

        act_event = threading.Event()
        deact_events = []

        def on_activate():
            activate_count[0] += 1

        def on_deactivate():
            deactivate_count[0] += 1

        mgr.set_callback(on_activate=on_activate, on_deactivate=on_deactivate)

        # Simulate 5 toggle cycles via internal _toggle() (no listener needed)
        for _ in range(5):
            with mgr._lock:
                mgr._toggle()
            time.sleep(0.05)   # let daemon thread run
            with mgr._lock:
                mgr._toggle()
            time.sleep(0.05)

        self.assertEqual(activate_count[0], 5)
        self.assertEqual(deactivate_count[0], 5)

    # 11. Hold mode: press → activate, release → deactivate
    def test_hold_mode(self):
        mgr = self._make_manager(mode="hold")
        activate_evt = threading.Event()
        deactivate_evt = threading.Event()

        mgr.set_callback(
            on_activate=lambda: activate_evt.set(),
            on_deactivate=lambda: deactivate_evt.set(),
        )

        # Simulate press
        with mgr._lock:
            if not mgr._hotkey_triggered:
                mgr._hotkey_triggered = True
                mgr._fire_activate()

        self.assertTrue(activate_evt.wait(timeout=1.0), "on_activate not called on press")
        self.assertTrue(mgr._is_active)

        # Simulate release
        with mgr._lock:
            mgr._hotkey_triggered = False
            if mgr._is_active:
                mgr._fire_deactivate()

        self.assertTrue(deactivate_evt.wait(timeout=1.0), "on_deactivate not called on release")
        self.assertFalse(mgr._is_active)

    # 12. Conflict detection: Ctrl+Alt+Delete → should be flagged as conflict
    def test_conflict_detection_known(self):
        """
        Ctrl+Alt+Delete is a known system shortcut.
        _parse_hotkey should produce ≥2 keys; ctrl+alt presence → conflict.
        """
        hotkey_str = "<ctrl>+<alt>+<delete>"
        parsed = _parse_hotkey(hotkey_str)

        # Should parse to at least 2 keys (ctrl + alt, delete may be unknown on stub)
        self.assertGreaterEqual(len(parsed), 2)

        # Conflict check: any ctrl key AND any alt key present → risky
        ctrl_keys = {Key["ctrl_l"], Key["ctrl_r"]}
        alt_keys = {Key["alt_l"], Key["alt_r"]}

        has_ctrl = bool(parsed & ctrl_keys)
        has_alt = bool(parsed & alt_keys)
        is_conflict = has_ctrl and has_alt

        self.assertTrue(is_conflict, "Ctrl+Alt+Delete must be detected as conflicting")

    # 13. Conflict detection: Ctrl+Shift+Space → safe (no alt)
    def test_conflict_detection_safe(self):
        hotkey_str = "<ctrl>+<shift>+space"
        parsed = _parse_hotkey(hotkey_str)

        self.assertGreaterEqual(len(parsed), 2)

        alt_keys = {Key["alt_l"], Key["alt_r"]}
        has_alt = bool(parsed & alt_keys)

        self.assertFalse(has_alt, "Ctrl+Shift+Space must not contain alt — not a conflict")

    # 14. MOUSE_BUTTON_MAP contains expected entries
    def test_mouse_button_map(self):
        for key in ("left", "right", "middle"):
            self.assertIn(key, _MOUSE_BUTTON_MAP, f"'{key}' missing from MOUSE_BUTTON_MAP")

        self.assertEqual(_MOUSE_BUTTON_MAP["left"], Button.left)
        self.assertEqual(_MOUSE_BUTTON_MAP["right"], Button.right)
        self.assertEqual(_MOUSE_BUTTON_MAP["middle"], Button.middle)

    # 15. _get_key_safe returns correct key or None
    def test_right_modifier_keys(self):
        result = _get_key_safe("ctrl_r")
        self.assertIsNotNone(result, "_get_key_safe('ctrl_r') should not return None")
        self.assertEqual(result, Key["ctrl_r"])

        # Non-existent key → None (no exception)
        result_bad = _get_key_safe("nonexistent_key_xyz")
        self.assertIsNone(result_bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
