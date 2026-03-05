"""Tests for src/autostart.py — cross-platform autostart management."""

from __future__ import annotations

import os
import sys
import types
import tempfile
import importlib
from pathlib import Path
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open

import pytest


# ---------------------------------------------------------------------------
# Helpers to import the module with a clean state
# ---------------------------------------------------------------------------


def _import_autostart():
    """Import (or reimport) src.autostart fresh."""
    import importlib
    import src.autostart as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# get_executable_path
# ---------------------------------------------------------------------------


class TestGetExecutablePath:
    def test_frozen_returns_sys_executable(self):
        """When running as a PyInstaller bundle, return sys.executable."""
        import src.autostart as mod
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", "/dist/WhisperVoice.exe"):
            result = mod.get_executable_path()
        assert result == "/dist/WhisperVoice.exe"

    def test_script_returns_abspath_of_argv0(self):
        """When running as a plain script, return the absolute path of sys.argv[0]."""
        import src.autostart as mod
        # Remove 'frozen' attribute if present
        if hasattr(sys, "frozen"):
            delattr(sys, "frozen")
        with patch.object(sys, "argv", ["whisper_voice/main.py"]):
            result = mod.get_executable_path()
        assert os.path.isabs(result)
        assert result.endswith("main.py")

    def test_not_frozen_when_no_frozen_attr(self):
        """Ensure we handle the normal case (no sys.frozen attribute)."""
        import src.autostart as mod
        # Ensure frozen is absent
        sys_frozen_backup = getattr(sys, "frozen", None)
        if hasattr(sys, "frozen"):
            delattr(sys, "frozen")
        try:
            with patch.object(sys, "argv", ["/app/main.py"]):
                path = mod.get_executable_path()
            assert os.path.isabs(path)
        finally:
            if sys_frozen_backup is not None:
                sys.frozen = sys_frozen_backup  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Linux implementation
# ---------------------------------------------------------------------------


class TestLinuxAutostart:
    """Tests for Linux XDG autostart. Patch os operations to avoid FS side effects."""

    def _get_mod(self):
        import src.autostart as mod
        return mod

    def test_linux_enable_creates_desktop_file(self):
        mod = self._get_mod()
        written_content = {}
        tmp_dir = tempfile.mkdtemp()
        desktop_path = os.path.join(tmp_dir, f"{mod.APP_NAME}.desktop")

        with patch("src.autostart._AUTOSTART_DIR", tmp_dir), \
             patch("src.autostart._DESKTOP_FILE", desktop_path), \
             patch("src.autostart.get_executable_path", return_value="/usr/bin/whisper-voice"):
            result = mod._linux_enable()

        assert result is True
        assert os.path.exists(desktop_path)
        content = Path(desktop_path).read_text(encoding="utf-8")
        assert "[Desktop Entry]" in content
        assert "Exec=/usr/bin/whisper-voice" in content
        assert f"Name={mod.APP_NAME}" in content
        assert "X-GNOME-Autostart-enabled=true" in content

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_linux_disable_removes_desktop_file(self):
        mod = self._get_mod()
        tmp_dir = tempfile.mkdtemp()
        desktop_path = os.path.join(tmp_dir, f"{mod.APP_NAME}.desktop")
        # Create the file first
        Path(desktop_path).write_text("[Desktop Entry]\n", encoding="utf-8")

        with patch("src.autostart._DESKTOP_FILE", desktop_path):
            result = mod._linux_disable()

        assert result is True
        assert not os.path.exists(desktop_path)

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_linux_disable_succeeds_when_file_absent(self):
        """Disabling when file doesn't exist should return True (idempotent)."""
        mod = self._get_mod()
        nonexistent = "/tmp/definitely_nonexistent_whisper_desktop_99999.desktop"
        with patch("src.autostart._DESKTOP_FILE", nonexistent):
            result = mod._linux_disable()
        assert result is True

    def test_linux_is_enabled_true_when_file_exists(self):
        mod = self._get_mod()
        tmp_dir = tempfile.mkdtemp()
        desktop_path = os.path.join(tmp_dir, f"{mod.APP_NAME}.desktop")
        Path(desktop_path).write_text("[Desktop Entry]\n", encoding="utf-8")

        with patch("src.autostart._DESKTOP_FILE", desktop_path):
            result = mod._linux_is_enabled()

        assert result is True

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_linux_is_enabled_false_when_file_absent(self):
        mod = self._get_mod()
        nonexistent = "/tmp/no_such_desktop_file_99999.desktop"
        with patch("src.autostart._DESKTOP_FILE", nonexistent):
            result = mod._linux_is_enabled()
        assert result is False


# ---------------------------------------------------------------------------
# Windows implementation (mocked winreg module)
# ---------------------------------------------------------------------------


class TestWindowsAutostart:
    """Windows tests: winreg is mocked so tests run on Linux too."""

    def _make_winreg_mock(self):
        """Build a minimal winreg mock."""
        winreg = MagicMock()
        winreg.HKEY_CURRENT_USER = 0x80000001
        winreg.KEY_SET_VALUE = 0x0002
        winreg.KEY_QUERY_VALUE = 0x0001
        winreg.REG_SZ = 1
        return winreg

    def test_windows_enable_calls_set_value(self):
        import src.autostart as mod
        winreg = self._make_winreg_mock()

        with patch.dict(sys.modules, {"winreg": winreg}), \
             patch("src.autostart.get_executable_path", return_value=r"C:\app\WhisperVoice.exe"):
            result = mod._windows_enable()

        assert result is True
        winreg.OpenKey.assert_called_once()
        winreg.SetValueEx.assert_called_once()
        # Verify the value contains the exe path
        args = winreg.SetValueEx.call_args[0]
        assert mod.APP_NAME in args
        assert r"C:\app\WhisperVoice.exe" in args[4]

    def test_windows_disable_calls_delete_value(self):
        import src.autostart as mod
        winreg = self._make_winreg_mock()

        with patch.dict(sys.modules, {"winreg": winreg}):
            result = mod._windows_disable()

        assert result is True
        winreg.DeleteValue.assert_called_once()

    def test_windows_disable_succeeds_when_not_found(self):
        """FileNotFoundError (key absent) should return True."""
        import src.autostart as mod
        winreg = self._make_winreg_mock()
        winreg.DeleteValue.side_effect = FileNotFoundError

        with patch.dict(sys.modules, {"winreg": winreg}):
            result = mod._windows_disable()

        assert result is True

    def test_windows_is_enabled_true_when_value_exists(self):
        import src.autostart as mod
        winreg = self._make_winreg_mock()
        # QueryValueEx returns a tuple (value, type)
        winreg.QueryValueEx.return_value = (r'"C:\app\WhisperVoice.exe"', 1)

        with patch.dict(sys.modules, {"winreg": winreg}):
            result = mod._windows_is_enabled()

        assert result is True

    def test_windows_is_enabled_false_when_key_missing(self):
        import src.autostart as mod
        winreg = self._make_winreg_mock()
        winreg.QueryValueEx.side_effect = FileNotFoundError

        with patch.dict(sys.modules, {"winreg": winreg}):
            result = mod._windows_is_enabled()

        assert result is False

    def test_windows_enable_returns_false_on_exception(self):
        import src.autostart as mod
        winreg = self._make_winreg_mock()
        winreg.OpenKey.side_effect = OSError("Access denied")

        with patch.dict(sys.modules, {"winreg": winreg}):
            result = mod._windows_enable()

        assert result is False


# ---------------------------------------------------------------------------
# Public API (sync_autostart, enable/disable/is_enabled)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_sync_autostart_true_calls_enable(self):
        import src.autostart as mod
        with patch.object(mod, "enable_autostart", return_value=True) as mock_enable, \
             patch.object(mod, "disable_autostart", return_value=True) as mock_disable:
            result = mod.sync_autostart(True)
        mock_enable.assert_called_once()
        mock_disable.assert_not_called()
        assert result is True

    def test_sync_autostart_false_calls_disable(self):
        import src.autostart as mod
        with patch.object(mod, "enable_autostart", return_value=True) as mock_enable, \
             patch.object(mod, "disable_autostart", return_value=True) as mock_disable:
            result = mod.sync_autostart(False)
        mock_disable.assert_called_once()
        mock_enable.assert_not_called()
        assert result is True


# ---------------------------------------------------------------------------
# Config integration: auto_start defaults to False
# ---------------------------------------------------------------------------


class TestConfigDefault:
    def test_auto_start_disabled_by_default(self):
        """DEFAULT_CONFIG must have auto_start=False."""
        from src.config import DEFAULT_CONFIG
        assert "auto_start" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["auto_start"] is False
