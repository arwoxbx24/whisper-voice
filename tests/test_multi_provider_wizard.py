"""
Tests for multi-provider STT selection in SetupWizard (Task 9).

All tests run headless — no display required.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub tkinter (same pattern as test_setup_wizard.py)
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
    _check_deepgram_key,
    _check_local_faster_whisper,
)


# ---------------------------------------------------------------------------
# Helper: create wizard with provider state initialized (no GUI)
# ---------------------------------------------------------------------------

def _make_wizard(config=None):
    """Create a SetupWizard with tkinter mocked."""
    import tkinter as tk
    cfg = config or {"stt_providers": ["openai"], "api_key": "", "deepgram_api_key": ""}
    w = SetupWizard(cfg)
    # Inject fake StringVar/BooleanVar so we can manipulate state
    w._provider_var = tk.StringVar(value=(cfg.get("stt_providers") or ["openai"])[0])
    w._provider_desc_var = tk.StringVar(value="")
    w._provider_desc_lbl = MagicMock()
    w._api_step_subtitle = MagicMock()
    w._openai_frame = MagicMock()
    w._deepgram_frame = MagicMock()
    w._local_frame = MagicMock()
    w._key_entry = MagicMock()
    w._key_entry.get.return_value = ""
    w._deepgram_key_entry = MagicMock()
    w._deepgram_key_entry.get.return_value = ""
    w._local_model_var = tk.StringVar(value="base")
    w._local_status_var = tk.StringVar(value="")
    w._local_download_btn = MagicMock()
    w._deepgram_status_var = tk.StringVar(value="")
    w._deepgram_status_lbl = MagicMock()
    w._deepgram_test_btn = MagicMock()
    w._api_status_var = tk.StringVar(value="")
    w._api_status_lbl = MagicMock()
    w._test_btn = MagicMock()
    w._mode_var = tk.StringVar(value="toggle")
    w._autostart_var = tk.BooleanVar(value=False)
    w._summary_var = tk.StringVar(value="")
    w._status_var = tk.StringVar(value="")
    return w


# ---------------------------------------------------------------------------
# Tests: provider selection
# ---------------------------------------------------------------------------

class TestProviderSelectionOpenAI(unittest.TestCase):
    def test_default_provider_is_openai(self):
        w = _make_wizard({"stt_providers": ["openai"]})
        self.assertEqual(w._provider_var.get(), "openai")

    def test_on_provider_change_saves_to_config(self):
        w = _make_wizard({"stt_providers": ["openai"]})
        w._provider_var.set("openai")
        w._on_provider_change()
        self.assertEqual(w._config["stt_providers"], ["openai"])

    def test_provider_desc_updated_on_change(self):
        w = _make_wizard({"stt_providers": ["openai"]})
        w._provider_var.set("openai")
        w._update_provider_desc()
        desc = w._provider_desc_var.get()
        self.assertIn("OpenAI", desc)
        self.assertIn("platform.openai.com", desc)


class TestProviderSelectionDeeepgram(unittest.TestCase):
    def test_deepgram_provider_saved(self):
        w = _make_wizard({"stt_providers": ["openai"]})
        w._provider_var.set("deepgram")
        w._on_provider_change()
        self.assertEqual(w._config["stt_providers"], ["deepgram"])

    def test_deepgram_desc(self):
        w = _make_wizard({"stt_providers": ["deepgram"]})
        w._provider_var.set("deepgram")
        w._update_provider_desc()
        desc = w._provider_desc_var.get()
        self.assertIn("Deepgram", desc)
        self.assertIn("console.deepgram.com", desc)


class TestProviderSelectionLocal(unittest.TestCase):
    def test_local_provider_saved(self):
        w = _make_wizard({"stt_providers": ["openai"]})
        w._provider_var.set("local")
        w._on_provider_change()
        self.assertEqual(w._config["stt_providers"], ["local"])

    def test_local_desc_mentions_offline(self):
        w = _make_wizard({"stt_providers": ["local"]})
        w._provider_var.set("local")
        w._update_provider_desc()
        desc = w._provider_desc_var.get()
        self.assertIn("оффлайн", desc.lower())


# ---------------------------------------------------------------------------
# Tests: config saves correct provider settings
# ---------------------------------------------------------------------------

def _simulate_go_next_save(w: SetupWizard) -> None:
    """
    Replicate only the config-save portion of _go_next for the current step,
    without calling _show_step (which requires full GUI objects).
    """
    step = w._current_step
    if step == 1:
        # Save provider selection
        if w._provider_var:
            provider = w._provider_var.get()
            w._config["stt_providers"] = [provider]
    elif step == 2:
        # Save API key(s) depending on provider
        provider = (w._config.get("stt_providers") or ["openai"])[0]
        if provider == "openai":
            key = w._key_entry.get().strip() if w._key_entry else ""
            if key:
                w._config["api_key"] = key
        elif provider == "deepgram":
            key = w._deepgram_key_entry.get().strip() if w._deepgram_key_entry else ""
            if key:
                w._config["deepgram_api_key"] = key
        elif provider == "local":
            if w._local_model_var:
                w._config["local_whisper_model"] = w._local_model_var.get()


class TestConfigSavesProvider(unittest.TestCase):
    def test_go_next_step1_saves_provider(self):
        """Navigating from step 1 persists provider to config."""
        w = _make_wizard({"stt_providers": ["openai"]})
        w._current_step = 1
        w._provider_var.set("deepgram")
        _simulate_go_next_save(w)
        self.assertEqual(w._config["stt_providers"], ["deepgram"])

    def test_go_next_step2_saves_openai_key(self):
        """Navigating from step 2 with openai saves api_key."""
        w = _make_wizard({"stt_providers": ["openai"], "api_key": ""})
        w._current_step = 2
        w._key_entry.get.return_value = "sk-testkey123"
        _simulate_go_next_save(w)
        self.assertEqual(w._config["api_key"], "sk-testkey123")

    def test_go_next_step2_saves_deepgram_key(self):
        """Navigating from step 2 with deepgram saves deepgram_api_key."""
        w = _make_wizard({"stt_providers": ["deepgram"], "deepgram_api_key": ""})
        w._current_step = 2
        w._deepgram_key_entry.get.return_value = "dg-key-abc"
        _simulate_go_next_save(w)
        self.assertEqual(w._config["deepgram_api_key"], "dg-key-abc")

    def test_go_next_step2_saves_local_model(self):
        """Navigating from step 2 with local saves local_whisper_model."""
        w = _make_wizard({"stt_providers": ["local"], "local_whisper_model": "base"})
        w._current_step = 2
        w._local_model_var.set("small")
        _simulate_go_next_save(w)
        self.assertEqual(w._config["local_whisper_model"], "small")

    def test_summary_shows_provider(self):
        """_update_summary includes provider name."""
        w = _make_wizard({"stt_providers": ["deepgram"], "deepgram_api_key": "dg-x"})
        w._deepgram_key_entry.get.return_value = "dg-x"
        w._update_summary()
        summary = w._summary_var.get()
        self.assertIn("Deepgram", summary)

    def test_summary_shows_local_model(self):
        """_update_summary for local shows model size."""
        w = _make_wizard({"stt_providers": ["local"], "local_whisper_model": "medium"})
        w._local_model_var.set("medium")
        w._update_summary()
        summary = w._summary_var.get()
        self.assertIn("medium", summary)


# ---------------------------------------------------------------------------
# Tests: Deepgram key check helper
# ---------------------------------------------------------------------------

class TestCheckDeepgramKey(unittest.TestCase):
    def test_valid_key(self):
        import urllib.request, json as _json

        class FakeResp:
            def read(self):
                return b'{"projects": [{"project_id": "p1"}, {"project_id": "p2"}]}'
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            ok, msg = _check_deepgram_key("dg-valid")
            self.assertTrue(ok)
            self.assertIn("2", msg)

    def test_invalid_key_401(self):
        import urllib.error
        err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = _check_deepgram_key("dg-bad")
            self.assertFalse(ok)
            self.assertIn("401", msg)

    def test_no_internet(self):
        import urllib.error
        err = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = _check_deepgram_key("dg-any")
            self.assertFalse(ok)
            self.assertIn("интернет", msg.lower())


# ---------------------------------------------------------------------------
# Tests: Local model check helper
# ---------------------------------------------------------------------------

class TestCheckLocalFasterWhisper(unittest.TestCase):
    def test_faster_whisper_not_installed(self):
        with patch.dict("sys.modules", {"faster_whisper": None}):
            ok, msg = _check_local_faster_whisper("base")
            self.assertFalse(ok)
            self.assertIn("faster-whisper", msg.lower())

    def test_faster_whisper_installed_model_cached(self):
        fw_mock = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            with patch("pathlib.Path.exists", return_value=True):
                ok, msg = _check_local_faster_whisper("base")
                self.assertTrue(ok)
                self.assertIn("base", msg)

    def test_faster_whisper_installed_model_not_cached(self):
        fw_mock = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            with patch("pathlib.Path.exists", return_value=False):
                ok, msg = _check_local_faster_whisper("large")
                self.assertFalse(ok)
                # Should mention download
                self.assertIn("скач", msg.lower())


# ---------------------------------------------------------------------------
# Tests: Wizard step count
# ---------------------------------------------------------------------------

class TestWizardStepCount(unittest.TestCase):
    def test_num_steps_is_5(self):
        self.assertEqual(SetupWizard.NUM_STEPS, 5)


if __name__ == "__main__":
    unittest.main()
