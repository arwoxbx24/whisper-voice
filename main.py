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
    _setup_log_dir()
    level = logging.DEBUG if debug else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        fh = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
        fh.setLevel(level)
        handlers.append(fh)
    except Exception:
        pass  # file handler unavailable (read-only FS etc.)

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
    """
    msg = (
        "Whisper Voice needs an OpenAI API key to work.\n\n"
        f"Config file location:\n  {config_path}\n\n"
        "Steps:\n"
        "1. Open the config file in Notepad\n"
        "2. Replace the empty \"api_key\" value with your OpenAI API key\n"
        "3. Save the file and restart Whisper Voice\n\n"
        "Get your API key at: https://platform.openai.com/api-keys"
    )
    _show_error_dialog("Whisper Voice — Setup Required", msg)


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

    # Check API key early and show friendly setup dialog instead of crashing
    providers = config.get("stt_providers", ["openai"])
    openai_key = config.get("api_key", "").strip()
    deepgram_key = config.get("deepgram_api_key", "").strip()
    has_local = "local" in providers
    needs_key = ("openai" in providers and not openai_key) or \
                ("deepgram" in providers and not deepgram_key)

    if not has_local and not openai_key and not deepgram_key:
        logger.warning("No API key configured — showing setup dialog")
        _show_config_setup_dialog(str(cfg_module.CONFIG_FILE))
        # Don't exit: if user dismisses dialog let app start anyway
        # (they can set key later via config file and restart)

    try:
        from src.app import WhisperVoiceApp
        app = WhisperVoiceApp(config)
        app.run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        _show_error_dialog("Whisper Voice — Error", str(exc))
        return 1
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unexpected error: %s\n%s", exc, tb)
        _show_error_dialog(
            "Whisper Voice — Unexpected Error",
            f"{exc}\n\nSee log for details:\n{_LOG_FILE}",
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
