"""
error_handler.py — Centralised error handling for Whisper Voice.

Provides:
  - setup_logging()         — RotatingFileHandler to ~/.whisper-voice/whisper-voice.log
  - categorize_error()      — maps exception to user-friendly Russian message
  - show_error_dialog()     — tkinter messagebox (thread-safe via tk root.after)
  - global_exception_handler() — sys.excepthook replacement

All user-facing messages are in Russian.
"""

import logging
import logging.handlers
import sys
import traceback
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Log file location
# ---------------------------------------------------------------------------

LOG_DIR = Path.home() / ".whisper-voice"
LOG_FILE = LOG_DIR / "whisper-voice.log"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(debug: bool = False) -> None:
    """
    Configure root logger with RotatingFileHandler (5 MB, 3 backups) and stderr.

    Safe to call multiple times — idempotent after first call.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        # Already configured — update level only
        level = logging.DEBUG if debug else logging.INFO
        root.setLevel(level)
        return

    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]

    try:
        rfh = logging.handlers.RotatingFileHandler(
            str(LOG_FILE),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        rfh.setLevel(level)
        rfh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(rfh)
    except Exception:
        pass  # Read-only FS or other issue — fall back to stderr only

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Error categorisation
# ---------------------------------------------------------------------------

_NETWORK_KEYWORDS = (
    "no internet", "urlopen error", "network", "connection refused",
    "connection reset", "connection aborted", "connection timed out",
    "name or service not known", "временно недоступен",
    "getaddrinfo failed", "socket", "errno 10060", "errno 10061",
    "errno 111",
)
_AUTH_KEYWORDS = (
    "401", "unauthorized", "invalid api key", "incorrect api key",
    "authentication", "invalid_api_key",
)
_RATE_LIMIT_KEYWORDS = ("429", "rate limit", "too many requests")
_AUDIO_KEYWORDS = (
    "microphone", "audio", "sounddevice", "portaudio", "input device",
    "no default input", "invalid input device",
)
_PERMISSION_KEYWORDS = (
    "permission", "access denied", "errno 13", "errno 5",
    "winerror 5",
)
_TIMEOUT_KEYWORDS = ("timeout", "timed out", "read timed out")
_DISK_KEYWORDS = ("disk", "no space left", "errno 28", "errno 122")


def categorize_error(exc: BaseException) -> str:
    """
    Convert a technical exception to a user-friendly Russian message.

    Args:
        exc: The caught exception.

    Returns:
        A localised string describing the problem and a suggested action.
    """
    raw = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Network / connectivity
    if (
        "connectionerror" in exc_type
        or "urlerror" in exc_type
        or "gaierror" in exc_type
        or any(kw in raw for kw in _NETWORK_KEYWORDS)
    ):
        return (
            "Нет подключения к интернету.\n"
            "Проверьте соединение и перезапустите приложение."
        )

    # Authentication / API key
    if (
        "authenticationerror" in exc_type
        or any(kw in raw for kw in _AUTH_KEYWORDS)
    ):
        return (
            "Неверный API ключ.\n\n"
            "Откройте Настройки через иконку в трее и введите корректный ключ.\n"
            "Получить ключ: https://platform.openai.com/api-keys"
        )

    # Rate limiting
    if any(kw in raw for kw in _RATE_LIMIT_KEYWORDS):
        return (
            "Превышен лимит запросов к API.\n"
            "Подождите немного и попробуйте снова."
        )

    # Audio / microphone
    if (
        "oserror" in exc_type
        and any(kw in raw for kw in _AUDIO_KEYWORDS)
    ) or any(kw in raw for kw in _AUDIO_KEYWORDS):
        return (
            "Микрофон не обнаружен или недоступен.\n"
            "Подключите микрофон и перезапустите приложение."
        )

    # File / device permission
    if (
        "permissionerror" in exc_type
        or any(kw in raw for kw in _PERMISSION_KEYWORDS)
    ):
        return (
            "Нет доступа к файлу или устройству.\n"
            "Закройте другие приложения, использующие микрофон, и повторите."
        )

    # Timeout
    if any(kw in raw for kw in _TIMEOUT_KEYWORDS):
        return (
            "Время ожидания истекло.\n"
            "Проверьте подключение к интернету."
        )

    # Disk space
    if any(kw in raw for kw in _DISK_KEYWORDS):
        return (
            "Недостаточно места на диске.\n"
            "Освободите место и перезапустите приложение."
        )

    # Generic fallback
    return (
        "Произошла непредвиденная ошибка.\n\n"
        f"Лог: {LOG_FILE}"
    )


# ---------------------------------------------------------------------------
# UI error dialog (thread-safe)
# ---------------------------------------------------------------------------


def show_error_dialog(
    title: str,
    message: str,
    details: Optional[str] = None,
    tk_root=None,
) -> None:
    """
    Show a user-friendly error dialog.

    Thread safety:
        If tk_root is provided the dialog is scheduled via root.after(0, ...)
        so it runs on the main tkinter thread.  Otherwise a standalone Tk
        window is created (safe from main thread only).

    Args:
        title:    Dialog window title.
        message:  Main message shown to the user (Russian).
        details:  Optional technical detail appended in smaller text.
        tk_root:  If provided, used to schedule the call on the main thread.
    """
    full_message = message
    if details:
        full_message = f"{message}\n\n[Детали]\n{details}"

    def _do_show():
        try:
            import tkinter as tk
            from tkinter import messagebox

            if tk_root is not None:
                # We're already on the main thread (called via root.after)
                messagebox.showerror(title, full_message, parent=tk_root)
            else:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showerror(title, full_message)
                root.destroy()
        except Exception:
            # Absolute fallback
            _write_error_to_log(title, full_message)

    if tk_root is not None:
        # Schedule on the tkinter main loop — safe from any thread
        try:
            tk_root.after(0, _do_show)
        except Exception:
            _do_show()
    else:
        _do_show()


def show_error_from_thread(
    title: str,
    message: str,
    tk_root=None,
) -> None:
    """
    Convenience wrapper: shows an error dialog safely from a background thread.

    Uses root.after() to dispatch to the tkinter main loop.
    If no root is available, falls back to standalone dialog.
    """
    show_error_dialog(title, message, tk_root=tk_root)


def _write_error_to_log(title: str, message: str) -> None:
    """Emergency fallback: write error to log file directly."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(str(LOG_FILE), "a", encoding="utf-8") as fh:
            fh.write(f"FATAL — {title}: {message}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Global exception hook
# ---------------------------------------------------------------------------


def global_exception_handler(exc_type, exc_value, exc_tb) -> None:
    """
    sys.excepthook replacement.

    Logs the full traceback and shows a user-friendly dialog.
    KeyboardInterrupt is re-raised normally.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical("Unhandled exception:\n%s", tb_str)

    friendly = categorize_error(exc_value)
    show_error_dialog(
        "Whisper Voice — Критическая ошибка",
        friendly,
        details=f"{exc_type.__name__}: {exc_value}\n\nЛог: {LOG_FILE}",
    )


def install_global_handler() -> None:
    """Install global_exception_handler as sys.excepthook."""
    sys.excepthook = global_exception_handler
    logger.debug("Global exception handler installed")
