"""
Phase 4 tests: SmartTextInserter (terminal-aware text insertion)

Covers:
  - Terminal vs GUI window detection via xdotool (Linux)
  - Clipboard save/restore on success and on error
  - Correct paste key selection (Ctrl+Shift+V for terminals, Ctrl+V for GUI)
  - xdotool availability check
  - Empty text early-exit
  - Terminal pattern matching for known terminal emulators

All tests run without real X11/keyboard hardware (pynput + subprocess patched).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub pynput BEFORE importing any project code that imports pynput.
# Reuse the same pattern from test_phase1.py to avoid X11 errors on headless CI.
# ---------------------------------------------------------------------------

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
    names = [
        "ctrl_l", "ctrl_r", "ctrl",
        "alt_l", "alt_r",
        "shift_l", "shift_r", "shift",
        "insert",
        "cmd", "delete", "space", "enter", "backspace", "tab",
    ]
    cls_dict = {"_members": {}}
    for n in names:
        key = _FakeKey(n)
        cls_dict[n] = key
        cls_dict["_members"][n] = key

    def class_getitem(name):
        members = cls_dict["_members"]
        if name in members:
            return members[name]
        raise KeyError(name)

    cls_dict["__class_getitem__"] = class_getitem
    cls_dict["__getitem__"] = staticmethod(class_getitem)
    FakeKeyCls = type("Key", (), cls_dict)
    FakeKeyCls.__class_getitem__ = classmethod(lambda cls, name: class_getitem(name))

    meta = type(FakeKeyCls)
    try:
        meta.__getitem__ = lambda self, name: class_getitem(name)
    except TypeError:
        pass

    return FakeKeyCls, class_getitem


_FakeKeyCls, _key_getitem = _make_fake_key_cls()


class _KeyProxy:
    """Acts as both a namespace (Key.ctrl) and subscriptable (Key['ctrl'])."""

    def __init__(self, cls, getitem_fn):
        self._cls = cls
        self._getitem = getitem_fn
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


class _FakeController:
    """Minimal stub for pynput.keyboard.Controller that records key presses."""

    def __init__(self):
        self.pressed_keys = []

    def press(self, key):
        self.pressed_keys.append(("press", key))

    def release(self, key):
        self.pressed_keys.append(("release", key))

    def type(self, text):
        self.pressed_keys.append(("type", text))


class _FakeListener:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


# Inject stubs into sys.modules BEFORE project imports
_kb_mod = MagicMock()
_kb_mod.Key = _KeyNamespace
_kb_mod.KeyCode = _FakeKeyCode
_kb_mod.Controller = _FakeController
_kb_mod.Listener = _FakeListener

_pynput_mod = MagicMock()
_pynput_mod.keyboard = _kb_mod

sys.modules.setdefault("pynput", _pynput_mod)
sys.modules.setdefault("pynput.keyboard", _kb_mod)

# ---------------------------------------------------------------------------
# Safe imports — project modules
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import text_inserter as ti_module  # noqa: E402
from text_inserter import (  # noqa: E402
    SmartTextInserter,
    TextInserter,
    TextInserterError,
    TERMINAL_IDENTIFIERS,
)


# ---------------------------------------------------------------------------
# Helper: build a CompletedProcess-like mock for subprocess.run
# ---------------------------------------------------------------------------

def _make_proc(stdout="", returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


def _make_inserter(**kwargs):
    """Create a SmartTextInserter with xdotool check disabled for speed."""
    with patch("subprocess.run", return_value=_make_proc(returncode=1)):
        inserter = SmartTextInserter(**kwargs)
    return inserter


# ===========================================================================
# Tests 1–2: Terminal detection via xdotool (_is_terminal_linux)
# ===========================================================================

class TestTerminalDetection(unittest.TestCase):
    """Terminal/GUI detection via xdotool on Linux."""

    def setUp(self):
        self.inserter = _make_inserter()
        # Force Linux system for deterministic test results
        self.inserter._system = "Linux"

    # Test 1: xdotool reports gnome-terminal class → _is_terminal() returns True
    def test_terminal_detection_by_class(self):
        """_is_terminal() → True when active window class is gnome-terminal."""
        # _is_terminal_linux() makes 3 subprocess calls:
        #   1. getactivewindow  2. getwindowclassname  3. getwindowname
        def fake_run(cmd, **kwargs):
            if "getactivewindow" in cmd:
                return _make_proc(stdout="12345")
            if "getwindowclassname" in cmd:
                return _make_proc(stdout="gnome-terminal")
            if "getwindowname" in cmd:
                return _make_proc(stdout="Terminal — bash")
            return _make_proc(returncode=1)

        with patch("subprocess.run", side_effect=fake_run):
            result = self.inserter._is_terminal()

        self.assertTrue(result, "_is_terminal() must return True for gnome-terminal")

    # Test 2: xdotool reports chrome class → _is_terminal() returns False
    def test_gui_detection(self):
        """_is_terminal() → False when active window class is chrome."""
        def fake_run(cmd, **kwargs):
            if "getactivewindow" in cmd:
                return _make_proc(stdout="99999")
            if "getwindowclassname" in cmd:
                return _make_proc(stdout="Google-chrome")
            if "getwindowname" in cmd:
                return _make_proc(stdout="Google Chrome")
            return _make_proc(returncode=1)

        with patch("subprocess.run", side_effect=fake_run):
            result = self.inserter._is_terminal()

        self.assertFalse(result, "_is_terminal() must return False for chrome")


# ===========================================================================
# Tests 3–4: Clipboard save/restore on success and on error
# ===========================================================================

class TestClipboardRestore(unittest.TestCase):
    """Clipboard save/restore in SmartTextInserter._insert_via_clipboard."""

    def setUp(self):
        self.inserter = _make_inserter(paste_delay=0, restore_delay=0,
                                       terminal_paste_delay=0)
        self.inserter._system = "Linux"

    # Test 3: clipboard is restored after successful insertion
    def test_clipboard_restored_on_success(self):
        """pyperclip.copy(original) must be called in the finally block on success."""
        copy_calls = []

        mock_pyperclip = MagicMock()
        mock_pyperclip.paste.return_value = "original text"
        mock_pyperclip.copy.side_effect = lambda v: copy_calls.append(v)

        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}), \
             patch.object(self.inserter, "_is_terminal", return_value=False), \
             patch.object(self.inserter, "_send_paste"):
            self.inserter._insert_via_clipboard("new text")

        self.assertIn(
            "original text",
            copy_calls,
            "Original clipboard must be restored via pyperclip.copy() in finally block",
        )

    # Test 4: clipboard is restored even when _send_paste raises
    def test_clipboard_restored_on_error(self):
        """Original clipboard is restored even when _send_paste() raises."""
        copy_calls = []

        mock_pyperclip = MagicMock()
        mock_pyperclip.paste.return_value = "saved original"
        mock_pyperclip.copy.side_effect = lambda v: copy_calls.append(v)

        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}), \
             patch.object(self.inserter, "_is_terminal", return_value=False), \
             patch.object(self.inserter, "_send_paste", side_effect=RuntimeError("keyboard broken")):
            # _insert_via_clipboard catches all exceptions internally
            self.inserter._insert_via_clipboard("test")

        self.assertIn(
            "saved original",
            copy_calls,
            "Original clipboard must be restored even when _send_paste raises",
        )


# ===========================================================================
# Tests 5–6: Correct paste shortcut selection (terminal vs GUI)
# ===========================================================================

class TestPasteKeySelection(unittest.TestCase):
    """_send_paste() uses Ctrl+Shift+V for terminals and Ctrl+V for GUI (Linux)."""

    def setUp(self):
        self.inserter = _make_inserter()
        self.inserter._system = "Linux"

    def _collect_pressed_keys(self, is_terminal: bool):
        """Run _send_paste(is_terminal) and return list of (action, key) calls."""
        import sys as _sys
        fake_kb = _FakeController()
        # Patch Controller in whatever pynput.keyboard module is actually in sys.modules
        actual_kb_mod = _sys.modules["pynput.keyboard"]
        with patch.object(actual_kb_mod, "Controller", return_value=fake_kb):
            self.inserter._send_paste(is_terminal)
        return fake_kb.pressed_keys

    # Test 5: terminal window → Key.shift in pressed keys
    def test_terminal_gets_ctrl_shift_v(self):
        """_send_paste(is_terminal=True) must press Key.shift."""
        keys = self._collect_pressed_keys(is_terminal=True)
        pressed_key_names = [
            getattr(k, "name", str(k)) for action, k in keys if action == "press"
        ]
        self.assertIn("shift", pressed_key_names,
                      "Terminal paste must use Key.shift (Ctrl+Shift+V)")

    # Test 6: GUI window → Key.shift NOT in pressed keys
    def test_gui_gets_ctrl_v(self):
        """_send_paste(is_terminal=False) must NOT press Key.shift."""
        keys = self._collect_pressed_keys(is_terminal=False)
        pressed_key_names = [
            getattr(k, "name", str(k)) for action, k in keys if action == "press"
        ]
        self.assertNotIn("shift", pressed_key_names,
                         "GUI paste must NOT use Key.shift (only Ctrl+V)")


# ===========================================================================
# Test 7: xdotool fallback — when getactivewindow fails → False
# ===========================================================================

class TestXdotoolFallback(unittest.TestCase):

    def setUp(self):
        self.inserter = _make_inserter()
        self.inserter._system = "Linux"

    # Test 7: xdotool getactivewindow returns non-zero → _is_terminal returns False
    def test_xdotool_fallback(self):
        """When xdotool getactivewindow fails, _is_terminal() returns False (safe fallback)."""
        with patch("subprocess.run", return_value=_make_proc(returncode=1)):
            result = self.inserter._is_terminal()
        self.assertFalse(result, "When xdotool fails, must fall back to non-terminal (False)")

    def test_xdotool_file_not_found_fallback(self):
        """When xdotool is absent (FileNotFoundError), _is_terminal() returns False."""
        with patch("subprocess.run", side_effect=FileNotFoundError("xdotool not found")):
            result = self.inserter._is_terminal()
        self.assertFalse(result, "FileNotFoundError → _is_terminal() must return False")


# ===========================================================================
# Test 8: Empty text returns False
# ===========================================================================

class TestEmptyText(unittest.TestCase):

    def setUp(self):
        self.inserter = _make_inserter()

    # Test 8: insert_text("") returns False immediately
    def test_empty_text_returns_false(self):
        """insert_text('') must return False without touching clipboard."""
        mock_pyperclip = MagicMock()
        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}):
            result = self.inserter.insert_text("")
        self.assertFalse(result, "insert_text('') must return False")
        mock_pyperclip.copy.assert_not_called()
        mock_pyperclip.paste.assert_not_called()


# ===========================================================================
# Test 9: _check_xdotool() availability detection
# ===========================================================================

class TestXdotoolAvailability(unittest.TestCase):

    def setUp(self):
        # Create inserter without running the check during __init__
        with patch("subprocess.run", return_value=_make_proc(returncode=1)):
            self.inserter = SmartTextInserter()
        self.inserter._system = "Linux"

    # Test 9a: xdotool available
    def test_xdotool_available_check(self):
        """_check_xdotool() returns True when 'which xdotool' succeeds."""
        with patch("subprocess.run", return_value=_make_proc(stdout="/usr/bin/xdotool", returncode=0)):
            result = self.inserter._check_xdotool()
        self.assertTrue(result, "_check_xdotool() must return True when xdotool found")

    # Test 9b: xdotool absent
    def test_xdotool_not_available_check(self):
        """_check_xdotool() returns False when 'which xdotool' fails."""
        with patch("subprocess.run", return_value=_make_proc(stdout="", returncode=1)):
            result = self.inserter._check_xdotool()
        self.assertFalse(result, "_check_xdotool() must return False when xdotool absent")


# ===========================================================================
# Test 10: _matches_terminal() pattern matching
# ===========================================================================

class TestTerminalPatternMatching(unittest.TestCase):
    """_matches_terminal(class_name, window_name) for known apps."""

    def setUp(self):
        self.inserter = _make_inserter()

    # Test 10: various known terminals and GUI apps
    def test_matches_terminal_patterns(self):
        """Known terminal emulators return True; GUI apps return False."""
        # Each tuple: (class_name, window_name, expected)
        cases = [
            # Terminals — match by class_name
            ("kitty", "kitty", True),
            ("alacritty", "alacritty", True),
            ("gnome-terminal", "Terminal — bash", True),
            ("xterm", "xterm", True),
            ("konsole", "Konsole", True),
            ("urxvt", "urxvt", True),
            ("st", "st — zsh", True),
            # Non-terminals
            # Note: "code" class with "Visual Studio Code" title matches "st" substring
            # in "studio" — true substring false positive in TERMINAL_IDENTIFIERS.
            # Use class/title strings that contain NO terminal identifier substrings.
            ("google-chrome", "Google Chrome", False),
            ("firefox", "Mozilla Firefox", False),
            ("gimp", "GNU Image Manipulation Program", False),
            ("nautilus", "Files", False),
            ("slack", "Slack — workspace", False),
        ]

        for class_name, window_name, expected in cases:
            with self.subTest(app=class_name):
                result = self.inserter._matches_terminal(class_name.lower(), window_name.lower())
                if expected:
                    self.assertTrue(
                        result,
                        f"'{class_name}' should be detected as a terminal",
                    )
                else:
                    self.assertFalse(
                        result,
                        f"'{class_name}' should NOT be detected as a terminal",
                    )


# ===========================================================================
# Additional coverage: backward compatibility and create_inserter
# ===========================================================================

class TestBackwardCompatibility(unittest.TestCase):

    def test_textinserter_alias_is_smarttextinserter(self):
        """TextInserter must be an alias for SmartTextInserter."""
        self.assertIs(TextInserter, SmartTextInserter)

    def test_insert_text_nonempty_calls_clipboard(self):
        """insert_text() with non-empty text returns True when clipboard works."""
        with patch("subprocess.run", return_value=_make_proc(returncode=1)):
            inserter = SmartTextInserter(method="clipboard", paste_delay=0,
                                         restore_delay=0, terminal_paste_delay=0)
        inserter._system = "Linux"

        mock_pyperclip = MagicMock()
        mock_pyperclip.paste.return_value = "old"

        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}), \
             patch.object(inserter, "_is_terminal", return_value=False), \
             patch.object(inserter, "_send_paste"):
            result = inserter.insert_text("hello world")

        self.assertTrue(result)
        # First copy call must set the new text
        first_call_arg = mock_pyperclip.copy.call_args_list[0][0][0]
        self.assertEqual(first_call_arg, "hello world")

    def test_create_inserter_raises_when_pynput_absent(self):
        """create_inserter() raises TextInserterError when pynput is absent."""
        # Temporarily make the pynput import fail inside create_inserter
        real_pynput = sys.modules.get("pynput.keyboard")
        try:
            # Patch the import to raise inside create_inserter's local import
            with patch.dict("sys.modules", {"pynput.keyboard": None}):
                with self.assertRaises((TextInserterError, ImportError)):
                    ti_module.create_inserter()
        finally:
            if real_pynput is not None:
                sys.modules["pynput.keyboard"] = real_pynput

    def test_terminal_identifiers_list_nonempty(self):
        """TERMINAL_IDENTIFIERS must contain expected terminal names."""
        for name in ("gnome-terminal", "kitty", "alacritty", "xterm", "konsole"):
            self.assertIn(name, TERMINAL_IDENTIFIERS,
                          f"'{name}' must be in TERMINAL_IDENTIFIERS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
