"""
text_inserter.py — Smart text insertion into the active application.

SmartTextInserter supports three strategies:
  - "xdotool": direct typing via xdotool (Linux only, no clipboard involved)
  - "clipboard": save/paste/restore via Ctrl+V or Ctrl+Shift+V for terminals
  - "auto": try xdotool first (if available), fall back to clipboard

Terminal detection (Linux/Windows/macOS) ensures Ctrl+Shift+V / Shift+Insert
is used when the active window is a terminal emulator.

Dependencies: pyperclip, pynput
"""

import logging
import os
import platform
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

TERMINAL_IDENTIFIERS = [
    "terminal", "console", "gnome-terminal", "konsole", "xterm", "rxvt",
    "kitty", "alacritty", "wezterm", "tmux", "screen", "st", "terminator",
    "tilix", "guake", "yakuake", "tilda", "hyper", "iterm", "iterm2",
    "powershell", "cmd.exe", "windowsterminal", "mintty", "putty",
    "nvim", "vim", "neovim", "emacs",
]


class TextInserterError(Exception):
    """Raised when text insertion fails and no fallback is available."""


class SmartTextInserter:
    """Inserts text at cursor position with terminal-aware paste shortcuts.

    Args:
        method: "auto" (default) tries xdotool first, falls back to clipboard.
                "clipboard" always uses clipboard paste.
                "xdotool" uses xdotool type only (Linux, no clipboard).
                "type" simulates individual keystrokes (legacy, slow).
        paste_delay: seconds between clipboard set and paste shortcut (default 0.06).
        terminal_paste_delay: extra wait after pasting in terminals (default 0.08).
        restore_delay: seconds before restoring original clipboard (default 0.12).
        fallback_to_type: if clipboard method fails, fall back to xdotool/type.
    """

    def __init__(self, method: str = "auto", paste_delay: float = 0.06,
                 terminal_paste_delay: float = 0.08, restore_delay: float = 0.12,
                 fallback_to_type: bool = True):
        self._method = method  # "auto", "clipboard", "xdotool", "type"
        self._paste_delay = paste_delay
        self._terminal_paste_delay = terminal_paste_delay
        self._restore_delay = restore_delay
        self._fallback_to_type = fallback_to_type
        self._system = platform.system()
        self._xdotool_available = self._check_xdotool()

        # Legacy attributes for backward compatibility
        self.method = method
        self.paste_delay = paste_delay
        self.restore_delay = restore_delay
        self.type_interval = 0.0

        try:
            from pynput.keyboard import Controller
            self._keyboard = Controller()
        except ImportError:
            self._keyboard = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_text(self, text: str) -> bool:
        """Insert text at cursor position. Returns True on success."""
        if not text:
            return False

        if self._method == "xdotool" or (self._method == "auto" and self._xdotool_available):
            if self._try_xdotool(text):
                return True
            if self._method == "xdotool":
                logger.warning("xdotool failed, no fallback (method=xdotool)")
                return False

        if self._method == "type":
            return self._insert_typing(text)

        # Clipboard method
        return self._insert_via_clipboard(text)

    def insert(self, text: str) -> None:
        """Legacy API: insert text, raises TextInserterError on failure."""
        if not text:
            return
        success = self.insert_text(text)
        if not success:
            raise TextInserterError("Text insertion failed via all available methods.")

    # ------------------------------------------------------------------
    # Private: xdotool
    # ------------------------------------------------------------------

    def _try_xdotool(self, text: str) -> bool:
        """Try inserting text via xdotool type. No clipboard involved."""
        try:
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    # ------------------------------------------------------------------
    # Private: clipboard
    # ------------------------------------------------------------------

    def _insert_via_clipboard(self, text: str) -> bool:
        """Insert text via clipboard with save/restore protection."""
        import pyperclip

        original_clipboard = None
        try:
            # Save original clipboard
            try:
                original_clipboard = pyperclip.paste()
            except Exception:
                original_clipboard = None

            # Set new text
            pyperclip.copy(text)
            time.sleep(self._paste_delay)

            # Determine if terminal
            is_terminal = self._is_terminal()

            # Send appropriate paste shortcut
            self._send_paste(is_terminal)

            if is_terminal:
                time.sleep(self._terminal_paste_delay)

            return True
        except Exception as e:
            logger.error(f"Clipboard insert failed: {e}")
            return False
        finally:
            # Restore clipboard
            time.sleep(self._restore_delay)
            try:
                if original_clipboard is not None:
                    pyperclip.copy(original_clipboard)
                else:
                    pyperclip.copy("")
            except Exception:
                pass

    def _send_paste(self, is_terminal: bool) -> None:
        """Send paste keyboard shortcut."""
        from pynput.keyboard import Controller, Key
        kb = Controller()

        if self._system == "Darwin":
            kb.press(Key.cmd)
            kb.press('v')
            kb.release('v')
            kb.release(Key.cmd)
        elif self._system == "Windows":
            if is_terminal:
                # Shift+Insert for Windows terminals
                kb.press(Key.shift)
                kb.press(Key.insert)
                kb.release(Key.insert)
                kb.release(Key.shift)
            else:
                kb.press(Key.ctrl)
                kb.press('v')
                kb.release('v')
                kb.release(Key.ctrl)
        else:
            # Linux
            if is_terminal:
                # Ctrl+Shift+V for terminals
                kb.press(Key.ctrl)
                kb.press(Key.shift)
                kb.press('v')
                kb.release('v')
                kb.release(Key.shift)
                kb.release(Key.ctrl)
            else:
                kb.press(Key.ctrl)
                kb.press('v')
                kb.release('v')
                kb.release(Key.ctrl)

    # ------------------------------------------------------------------
    # Private: legacy typing mode
    # ------------------------------------------------------------------

    def _insert_typing(self, text: str) -> bool:
        """Type text character by character using pynput (legacy, slow)."""
        if self._keyboard is None:
            logger.error("pynput not available for typing mode")
            return False
        try:
            from pynput.keyboard import KeyCode
            for char in text:
                try:
                    self._keyboard.type(char)
                except Exception:
                    try:
                        kc = KeyCode.from_char(char)
                        self._keyboard.press(kc)
                        self._keyboard.release(kc)
                    except Exception as exc:
                        logger.debug("Skipping untyped char %r: %s", char, exc)
                if self.type_interval:
                    time.sleep(self.type_interval)
            return True
        except Exception as e:
            logger.error(f"Type insert failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def _is_terminal(self) -> bool:
        """Detect if active window is a terminal."""
        try:
            if self._system == "Linux":
                return self._is_terminal_linux()
            elif self._system == "Windows":
                return self._is_terminal_windows()
            elif self._system == "Darwin":
                return self._is_terminal_macos()
        except Exception:
            pass
        return False

    def _is_terminal_linux(self) -> bool:
        """Linux terminal detection via xdotool."""
        try:
            wid = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2
            )
            if wid.returncode != 0:
                return False
            window_id = wid.stdout.strip()

            # Get window class name
            cls = subprocess.run(
                ["xdotool", "getwindowclassname", window_id],
                capture_output=True, text=True, timeout=2
            )
            class_name = cls.stdout.strip().lower() if cls.returncode == 0 else ""

            # Get window name
            name_result = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                capture_output=True, text=True, timeout=2
            )
            window_name = name_result.stdout.strip().lower() if name_result.returncode == 0 else ""

            return self._matches_terminal(class_name, window_name)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _is_terminal_windows(self) -> bool:
        """Windows terminal detection via ctypes."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd) + 1
            buf = ctypes.create_unicode_buffer(length)
            user32.GetWindowTextW(hwnd, buf, length)
            window_title = buf.value.lower()
            return self._matches_terminal("", window_title)
        except Exception:
            return False

    def _is_terminal_macos(self) -> bool:
        """macOS terminal detection via osascript."""
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                app_name = result.stdout.strip().lower()
                return self._matches_terminal(app_name, app_name)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _matches_terminal(self, class_name: str, window_name: str) -> bool:
        """Check if window class or name matches known terminals."""
        combined = f"{class_name} {window_name}"
        return any(t in combined for t in TERMINAL_IDENTIFIERS)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_xdotool(self) -> bool:
        """Check if xdotool is available."""
        if self._system != "Linux":
            return False
        try:
            result = subprocess.run(
                ["which", "xdotool"], capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _paste_key(self):
        """Return the modifier key used for paste (legacy compatibility)."""
        from pynput.keyboard import Key
        if self._system == "Darwin":
            return Key.cmd
        return Key.ctrl

    def _safe_get_clipboard(self) -> Optional[str]:
        """Return current clipboard text, or None on failure (legacy)."""
        try:
            import pyperclip
            return pyperclip.paste()
        except Exception as exc:
            logger.debug("Could not read clipboard: %s", exc)
            return None

    def _safe_set_clipboard(self, text: str) -> None:
        """Set clipboard to text, ignoring errors (legacy)."""
        try:
            import pyperclip
            pyperclip.copy(text)
        except Exception as exc:
            logger.debug("Could not restore clipboard: %s", exc)


# Backward compatibility alias
TextInserter = SmartTextInserter


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def create_inserter(prefer_clipboard: bool = True, **kwargs) -> SmartTextInserter:
    """Return a SmartTextInserter instance.

    Falls back to type mode automatically when pyperclip/pynput are absent.
    """
    try:
        import pyperclip
        _pyperclip_ok = True
    except ImportError:
        _pyperclip_ok = False

    try:
        from pynput.keyboard import Controller
        _pynput_ok = True
    except ImportError:
        _pynput_ok = False

    if not _pynput_ok:
        raise TextInserterError(
            "pynput is not installed. Run: pip install pynput"
        )

    if prefer_clipboard and _pyperclip_ok:
        return SmartTextInserter(method="auto", **kwargs)
    else:
        return SmartTextInserter(method="type", **kwargs)


# ------------------------------------------------------------------
# Quick smoke-test (run directly: python text_inserter.py)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    sample = "Hello from SmartTextInserter!"

    print(f"Inserting via method={mode!r} in 3 seconds — focus a text field...")
    time.sleep(3)

    inserter = SmartTextInserter(method=mode)
    inserter.insert_text(sample)
    print("Done.")
