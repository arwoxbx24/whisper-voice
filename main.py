"""
Whisper Voice — entry point.

Usage:
    python main.py [--config PATH] [--debug]
"""

import argparse
import logging
import sys
from pathlib import Path


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


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = _parse_args()
    _setup_logging(args.debug)

    logger = logging.getLogger("whisper_voice")

    # Allow overriding the config file path via --config
    if args.config:
        import src.config as cfg_module
        from pathlib import Path
        cfg_module.CONFIG_FILE = Path(args.config).expanduser().resolve()
        cfg_module.CONFIG_DIR = cfg_module.CONFIG_FILE.parent

    try:
        from src import config as cfg_module
        config = cfg_module.load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    try:
        from src.app import WhisperVoiceApp
        app = WhisperVoiceApp(config)
        app.run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
