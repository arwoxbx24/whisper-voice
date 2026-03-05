"""
Whisper Voice — entry point.

Usage:
    python main.py [--config PATH] [--debug]
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Log directory / file — set up FIRST, before any imports that might fail
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".whisper-voice"
_LOG_FILE = _LOG_DIR / "whisper-voice.log"


def _setup_log_dir() -> None:
    """Create the log directory early so we can write crash logs."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # If we can't create the dir, logging will fall back to stderr


def _setup_logging(debug: bool) -> None:
    """Setup logging with RotatingFileHandler (5 MB, 3 backups) via error_handler."""
    _setup_log_dir()
    try:
        from src.error_handler import setup_logging, install_global_handler
        setup_logging(debug=debug)
        install_global_handler()
    except Exception:
        # Fallback if error_handler not available
        level = logging.DEBUG if debug else logging.INFO
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
        try:
            import logging.handlers
            rfh = logging.handlers.RotatingFileHandler(
                str(_LOG_FILE), maxBytes=5 * 1024 * 1024,
                backupCount=3, encoding="utf-8",
            )
            handlers.append(rfh)
        except Exception:
            try:
                fh = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
                handlers.append(fh)
            except Exception:
                pass
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )


def _show_error_dialog(title: str, message: str) -> None:
    """
    Show an error message box.  Used when running as a --windowed exe so
    the user can see fatal errors even without a console.
    Silently falls back to stderr if tkinter is unavailable.
    """
    try:
        from src.error_handler import show_error_dialog
        show_error_dialog(title, message)
        return
    except Exception:
        pass
    # Direct fallback without error_handler
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        # Absolute fallback: write to log file
        try:
            with open(str(_LOG_FILE), "a", encoding="utf-8") as fh:
                fh.write(f"FATAL — {title}: {message}\n")
        except Exception:
            pass


def _show_config_setup_dialog(config_path: str) -> None:
    """
    Show a friendly guide to set up the API key when no key is configured.
    Tries to use the SetupWizard first; falls back to a simple error dialog.
    """
    try:
        from src.setup_wizard import SetupWizard
        from src import config as cfg_module
        config = cfg_module.load_config()
        wizard = SetupWizard(config)
        wizard.run()
        return
    except Exception:
        pass
    msg = (
        "Whisper Voice нужен API ключ OpenAI.\n\n"
        f"Файл настроек:\n  {config_path}\n\n"
        "Шаги:\n"
        "1. Откройте файл настроек в Блокноте\n"
        "2. Вставьте ваш API ключ в поле \"api_key\"\n"
        "3. Сохраните и перезапустите Whisper Voice\n\n"
        "Получить ключ: https://platform.openai.com/api-keys"
    )
    _show_error_dialog("Whisper Voice — Требуется настройка", msg)


def _run_setup_wizard(config: dict, on_complete=None) -> dict:
    """
    Run the Setup Wizard and return the (possibly updated) config.
    Called on first launch (no api_key configured).
    """
    try:
        from src.setup_wizard import SetupWizard
        from src import config as cfg_module

        updated_config = {}

        def _on_save(new_cfg):
            updated_config.update(new_cfg)

        wizard = SetupWizard(config, on_save=_on_save)
        wizard.run()

        if updated_config:
            return updated_config
    except Exception as exc:
        logger = logging.getLogger("whisper_voice")
        logger.warning("SetupWizard failed: %s", exc)

    return config


def _friendly_error(exc_or_str) -> str:
    """Convert exception or error string to user-friendly Russian description."""
    try:
        from src.error_handler import categorize_error
        if isinstance(exc_or_str, BaseException):
            return categorize_error(exc_or_str)
        # Wrap string in a generic Exception for categorize_error
        return categorize_error(Exception(exc_or_str))
    except Exception:
        pass
    # Inline fallback if error_handler not importable
    raw = str(exc_or_str)
    raw_lower = raw.lower()
    if "no internet" in raw_lower or "urlopen error" in raw_lower or "network" in raw_lower:
        return "Нет подключения к интернету.\nПроверьте соединение и перезапустите приложение."
    if "401" in raw or "unauthorized" in raw_lower or "invalid api key" in raw_lower:
        return (
            "Неверный API ключ.\n\n"
            "Откройте Настройки через иконку в трее и введите корректный ключ.\n"
            "Получить ключ: https://platform.openai.com/api-keys"
        )
    if "429" in raw or "rate limit" in raw_lower:
        return "Превышен лимит запросов к OpenAI API.\nПодождите немного и попробуйте снова."
    if "microphone" in raw_lower or "audio" in raw_lower or "sounddevice" in raw_lower:
        return "Микрофон не обнаружен или недоступен.\nПроверьте подключение микрофона."
    if "permission" in raw_lower or "access denied" in raw_lower:
        return "Нет доступа к устройству.\nПроверьте права доступа к микрофону."
    if "timeout" in raw_lower:
        return "Время ожидания истекло.\nПроверьте подключение к интернету."
    return f"Произошла непредвиденная ошибка.\n\nЛог: {_LOG_FILE}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="whisper-voice",
        description="Real-time speech-to-text via OpenAI Whisper API",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to config.json (default: ~/.whisper-voice/config.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    # Parse args first so --debug works even if everything else fails
    try:
        args = _parse_args()
    except SystemExit:
        raise
    except Exception:
        args_debug = "--debug" in sys.argv
        args = argparse.Namespace(config=None, debug=args_debug)

    _setup_logging(getattr(args, "debug", False))
    logger = logging.getLogger("whisper_voice")
    logger.info("Whisper Voice starting (Python %s)", sys.version.split()[0])
    logger.info("Log file: %s", _LOG_FILE)

    # Allow overriding the config file path via --config
    if getattr(args, "config", None):
        try:
            import src.config as cfg_module
            cfg_module.CONFIG_FILE = Path(args.config).expanduser().resolve()
            cfg_module.CONFIG_DIR = cfg_module.CONFIG_FILE.parent
        except Exception as exc:
            logger.error("Failed to override config path: %s", exc)

    try:
        from src import config as cfg_module
        config = cfg_module.load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        _show_error_dialog(
            "Whisper Voice — Config Error",
            f"Failed to load configuration:\n{exc}\n\n"
            f"Check the log file for details:\n{_LOG_FILE}",
        )
        return 1

    # Check API key early — show Setup Wizard on first run
    providers = config.get("stt_providers", ["openai"])
    openai_key = config.get("api_key", "").strip()
    deepgram_key = config.get("deepgram_api_key", "").strip()
    has_local = "local" in providers
    is_first_run = not cfg_module.CONFIG_FILE.exists() or (
        not openai_key and not deepgram_key and not has_local
    )

    if not has_local and not openai_key and not deepgram_key:
        logger.warning("No API key configured — launching Setup Wizard")
        config = _run_setup_wizard(config)
        # Reload to pick up any changes saved by wizard
        try:
            config = cfg_module.load_config()
        except Exception:
            pass

    try:
        from src.app import WhisperVoiceApp
        app = WhisperVoiceApp(config)
        app.run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        _show_error_dialog("Whisper Voice — Ошибка", _friendly_error(exc))
        return 1
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unexpected error: %s\n%s", exc, tb)
        _show_error_dialog(
            "Whisper Voice — Неожиданная ошибка",
            f"{_friendly_error(exc)}\n\nЛог-файл:\n{_LOG_FILE}",
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
