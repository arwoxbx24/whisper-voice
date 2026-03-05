"""
Tests for SetupWizard (setup_wizard.py).

All tests run headless (no display required) by testing only
the non-GUI helper functions and logic.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub tkinter so tests run without a display
# ---------------------------------------------------------------------------

def _stub_tkinter():
    if "tkinter" in sys.modules:
        return

    tk_mock = types.ModuleType("tkinter")

    class FakeStringVar:
        def __init__(self, value=""):
            self._v = value
        def get(self):
            return self._v
        def set(self, v):
            self._v = v

    class FakeBooleanVar:
        def __init__(self, value=False):
            self._v = value
        def get(self):
            return self._v
        def set(self, v):
            self._v = v

    tk_mock.Tk = MagicMock()
    tk_mock.Toplevel = MagicMock()
    tk_mock.Frame = MagicMock()
    tk_mock.Label = MagicMock()
    tk_mock.Button = MagicMock()
    tk_mock.Entry = MagicMock()
    tk_mock.Text = MagicMock()
    tk_mock.Checkbutton = MagicMock()
    tk_mock.Radiobutton = MagicMock()
    tk_mock.StringVar = FakeStringVar
    tk_mock.BooleanVar = FakeBooleanVar
    tk_mock.messagebox = MagicMock()

    sys.modules["tkinter"] = tk_mock
    sys.modules["tkinter.ttk"] = types.ModuleType("tkinter.ttk")
    sys.modules["tkinter.font"] = types.ModuleType("tkinter.font")


_stub_tkinter()


# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

from src.setup_wizard import (
    SetupWizard,
    _check_openai_key,
    _normalize_tk_key,
    _build_pynput_hotkey,
    _format_hotkey_display_fn,
    APP_VERSION,
)


# Helper alias
def _format(hotkey: str) -> str:
    return _format_hotkey_display_fn(hotkey)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFormatHotkeyDisplay(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(_format("<ctrl>+<shift>+space"), "Ctrl + Shift + Space")

    def test_single_key(self):
        self.assertEqual(_format("<f5>"), "F5")

    def test_empty(self):
        self.assertEqual(_format(""), "Не задана")

    def test_no_brackets(self):
        # bare tokens should just be capitalized
        result = _format("ctrl+shift+space")
        self.assertIn("Ctrl", result)
        self.assertIn("Shift", result)


class TestBuildPynputHotkey(unittest.TestCase):
    def test_ctrl_shift_space(self):
        keys = {"Ctrl", "Shift", "Space"}
        result = _build_pynput_hotkey(keys)
        self.assertIn("<ctrl>", result)
        self.assertIn("<shift>", result)
        self.assertIn("space", result)

    def test_empty_keys(self):
        result = _build_pynput_hotkey(set())
        # Falls back to default
        self.assertEqual(result, "<ctrl>+<shift>+space")

    def test_modifier_order(self):
        keys = {"Shift", "Ctrl", "A"}
        result = _build_pynput_hotkey(keys)
        # Ctrl must appear before Shift
        self.assertLess(result.index("<ctrl>"), result.index("<shift>"))


class TestNormalizeTkKey(unittest.TestCase):
    def _make_event(self, keysym: str):
        event = MagicMock()
        event.keysym = keysym
        return event

    def test_control_l(self):
        e = self._make_event("Control_L")
        self.assertEqual(_normalize_tk_key(e), "Ctrl")

    def test_shift_r(self):
        e = self._make_event("Shift_R")
        self.assertEqual(_normalize_tk_key(e), "Shift")

    def test_space(self):
        e = self._make_event("space")
        self.assertEqual(_normalize_tk_key(e), "Space")

    def test_letter(self):
        e = self._make_event("a")
        self.assertEqual(_normalize_tk_key(e), "A")

    def test_f5(self):
        e = self._make_event("F5")
        self.assertEqual(_normalize_tk_key(e), "F5")


class TestCheckOpenAIKey(unittest.TestCase):
    def test_valid_key(self):
        import urllib.request

        class FakeResponse:
            def __init__(self):
                self.data = b'{"data": [{"id": "whisper-1"}, {"id": "gpt-4"}]}'
            def read(self):
                return self.data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            ok, msg = _check_openai_key("sk-test123")
            self.assertTrue(ok)
            self.assertIn("2", msg)  # 2 models

    def test_invalid_key_401(self):
        import urllib.error

        err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = _check_openai_key("sk-bad")
            self.assertFalse(ok)
            self.assertIn("401", msg)

    def test_rate_limit_429(self):
        import urllib.error

        err = urllib.error.HTTPError(url="", code=429, msg="Too Many Requests", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = _check_openai_key("sk-ok")
            # 429 = key is valid but rate-limited
            self.assertTrue(ok)

    def test_no_internet(self):
        import urllib.error

        err = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = _check_openai_key("sk-anything")
            self.assertFalse(ok)
            self.assertIn("интернет", msg.lower())


class TestSetupWizardInit(unittest.TestCase):
    def test_config_deep_copied(self):
        config = {"api_key": "sk-original", "hotkey_mode": "toggle"}
        wizard = SetupWizard(config)
        # Modifying original should not affect wizard's config
        config["api_key"] = "sk-changed"
        self.assertEqual(wizard._config["api_key"], "sk-original")

    def test_defaults(self):
        wizard = SetupWizard({})
        self.assertEqual(wizard._current_step, 0)
        self.assertIsNone(wizard._root)
        self.assertFalse(wizard._recording_hotkey)

    def test_on_save_callback(self):
        saved = {}

        def _on_save(cfg):
            saved.update(cfg)

        wizard = SetupWizard({"api_key": "sk-x"}, on_save=_on_save)
        # Simulate saving: on_save should be called with config
        wizard._config = {"api_key": "sk-new", "hotkey": "<ctrl>+<shift>+space", "hotkey_mode": "toggle"}

        # Directly invoke on_save as _finish would
        wizard._on_save(wizard._config)
        self.assertEqual(saved.get("api_key"), "sk-new")


class TestAppVersion(unittest.TestCase):
    def test_version_is_string(self):
        self.assertIsInstance(APP_VERSION, str)
        self.assertTrue(len(APP_VERSION) > 0)


if __name__ == "__main__":
    unittest.main()
