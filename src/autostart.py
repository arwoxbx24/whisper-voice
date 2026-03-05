"""Cross-platform autostart management.

Registers (or removes) the application from the OS startup sequence
so it launches automatically when the user logs in.

Windows: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run  (no UAC needed)
Linux:   ~/.config/autostart/<APP_NAME>.desktop  (XDG autostart spec)
macOS:   not supported (returns False silently)
"""

from __future__ import annotations

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

APP_NAME = "WhisperVoice"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_executable_path() -> str:
    """Return the path to the running executable.

    Handles two cases:
    - PyInstaller bundle  (sys.frozen=True): returns sys.executable (the .exe)
    - Plain Python script:                  returns the absolute path of sys.argv[0]
    """
    if getattr(sys, "frozen", False):
        # PyInstaller bundles the app: sys.executable is the .exe
        return sys.executable
    else:
        # Normal Python run: argv[0] is the script path
        return os.path.abspath(sys.argv[0])


def is_autostart_enabled() -> bool:
    """Return True if autostart entry exists for the current platform."""
    system = platform.system()
    if system == "Windows":
        return _windows_is_enabled()
    elif system == "Linux":
        return _linux_is_enabled()
    return False


def enable_autostart() -> bool:
    """Register the app to start at login. Returns True on success."""
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_enable()
        elif system == "Linux":
            return _linux_enable()
    except Exception as exc:
        logger.error("Failed to enable autostart: %s", exc)
    return False


def disable_autostart() -> bool:
    """Remove the app from startup registry/folder. Returns True on success."""
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_disable()
        elif system == "Linux":
            return _linux_disable()
    except Exception as exc:
        logger.error("Failed to disable autostart: %s", exc)
    return False


def sync_autostart(enabled: bool) -> bool:
    """Enable or disable autostart according to *enabled* flag.

    Convenience wrapper used when config changes.
    Returns True on success, False on failure.
    """
    if enabled:
        return enable_autostart()
    else:
        return disable_autostart()


# ---------------------------------------------------------------------------
# Windows implementation
# ---------------------------------------------------------------------------


def _windows_enable() -> bool:
    import winreg  # noqa: PLC0415 — Windows-only, imported inside function

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    exe_path = get_executable_path()
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{exe_path}"')
        winreg.CloseKey(key)
        logger.info("Autostart enabled (Windows): %s", exe_path)
        return True
    except Exception as exc:
        logger.error("Windows autostart enable failed: %s", exc)
        return False


def _windows_disable() -> bool:
    import winreg  # noqa: PLC0415

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, APP_NAME)
        winreg.CloseKey(key)
        logger.info("Autostart disabled (Windows)")
        return True
    except FileNotFoundError:
        # Value did not exist — that is the desired state, so succeed
        return True
    except Exception as exc:
        logger.error("Windows autostart disable failed: %s", exc)
        return False


def _windows_is_enabled() -> bool:
    import winreg  # noqa: PLC0415

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_QUERY_VALUE
        )
        winreg.QueryValueEx(key, APP_NAME)
        winreg.CloseKey(key)
        return True
    except (FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Linux implementation (XDG autostart)
# ---------------------------------------------------------------------------

_AUTOSTART_DIR = os.path.expanduser("~/.config/autostart")
_DESKTOP_FILE = os.path.join(_AUTOSTART_DIR, f"{APP_NAME}.desktop")


def _linux_enable() -> bool:
    os.makedirs(_AUTOSTART_DIR, exist_ok=True)
    exe_path = get_executable_path()
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        f"Exec={exe_path}\n"
        "Hidden=false\n"
        "NoDisplay=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    with open(_DESKTOP_FILE, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("Autostart enabled (Linux): %s", _DESKTOP_FILE)
    return True


def _linux_disable() -> bool:
    if os.path.exists(_DESKTOP_FILE):
        os.remove(_DESKTOP_FILE)
        logger.info("Autostart disabled (Linux)")
    return True


def _linux_is_enabled() -> bool:
    return os.path.exists(_DESKTOP_FILE)
