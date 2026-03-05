"""
Config management for Whisper Voice.

Loads/saves config from ~/.whisper-voice/config.json.
Auto-creates the config directory if it does not exist.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".whisper-voice"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    # Core
    "api_key": "",
    "language": "ru",
    # Hotkey
    "hotkey": "<ctrl>+<shift>+space",
    "hotkey_mode": "toggle",          # "toggle" | "hold"
    "mouse_button": None,
    # STT Providers (priority order)
    "stt_providers": ["openai"],
    "deepgram_api_key": "",
    "local_whisper_model": "base",
    "local_whisper_device": "cpu",
    "model": "whisper-1",
    "prompt_context": (
        "Расшифровка речи. Технические термины: Python, JavaScript, Docker, "
        "Kubernetes, API, REST, GraphQL, SQL, Git, CI/CD."
    ),
    # Text insertion
    "insert_method": "auto",
    # Audio cache
    "audio_cache_enabled": True,
    "audio_cache_max_retries": 5,
    # UX
    "sound_feedback": True,
    "auto_start": False,
    "log_level": "INFO",
}

VALID_HOTKEY_MODES = {"toggle", "hold"}
VALID_INSERT_METHODS = {"auto", "clipboard", "xdotool", "type"}
VALID_STT_PROVIDERS = {"openai", "deepgram", "local"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}

_MODIFIER_NAMES = {"ctrl", "shift", "alt", "cmd"}


def normalize_hotkey(s: str) -> str:
    """Wrap bare modifier names in angle brackets.

    E.g. "ctrl+shift+space" → "<ctrl>+<shift>+space"
    Already-wrapped tokens like "<ctrl>" are left unchanged.
    """
    parts = s.split("+")
    result = []
    for part in parts:
        stripped = part.strip()
        if stripped.lower() in _MODIFIER_NAMES and not (stripped.startswith("<") and stripped.endswith(">")):
            result.append(f"<{stripped.lower()}>")
        else:
            result.append(stripped)
    return "+".join(result)


class ConfigError(Exception):
    """Raised when config validation fails."""


def _ensure_dir() -> None:
    """Create ~/.whisper-voice/ if it does not exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _validate(cfg: dict[str, Any]) -> None:
    """Validate config values. Raises ConfigError on bad values."""
    if not isinstance(cfg.get("api_key", ""), str):
        raise ConfigError("api_key must be a string")

    hotkey_mode = cfg.get("hotkey_mode", "toggle")
    if hotkey_mode not in VALID_HOTKEY_MODES:
        raise ConfigError(
            f"hotkey_mode must be one of {VALID_HOTKEY_MODES}, got {hotkey_mode!r}"
        )

    insert_method = cfg.get("insert_method", "auto")
    if insert_method not in VALID_INSERT_METHODS:
        raise ConfigError(
            f"insert_method must be one of {VALID_INSERT_METHODS}, got {insert_method!r}"
        )

    stt_providers = cfg.get("stt_providers", ["openai"])
    if not isinstance(stt_providers, list):
        raise ConfigError("stt_providers must be a list")
    for prov in stt_providers:
        if prov not in VALID_STT_PROVIDERS:
            raise ConfigError(
                f"stt_providers element must be one of {VALID_STT_PROVIDERS}, got {prov!r}"
            )

    log_level = cfg.get("log_level", "INFO")
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(
            f"log_level must be one of {VALID_LOG_LEVELS}, got {log_level!r}"
        )



def load_config() -> dict[str, Any]:
    """Load config from disk, creating defaults if the file does not exist.

    Returns:
        Merged config dict (defaults + user values).

    Raises:
        ConfigError: if the saved config fails validation.
    """
    _ensure_dir()

    if not CONFIG_FILE.exists():
        logger.info("Config file not found, creating default at %s", CONFIG_FILE)
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as fh:
            user_cfg: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Config file is corrupted (%s), using defaults", exc)
        return dict(DEFAULT_CONFIG)

    # Merge: defaults first, then user values (handles new keys in future versions)
    merged = {**DEFAULT_CONFIG, **user_cfg}
    # Normalize hotkey format so both "ctrl+shift+space" and "<ctrl>+<shift>+space" work
    merged["hotkey"] = normalize_hotkey(merged["hotkey"])
    _validate(merged)
    return merged


def save_config(cfg: dict[str, Any]) -> None:
    """Persist config to ~/.whisper-voice/config.json.

    Args:
        cfg: Config dict to save.

    Raises:
        ConfigError: if cfg fails validation.
    """
    _ensure_dir()
    _validate(cfg)
    tmp_path = CONFIG_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp_path.replace(CONFIG_FILE)
    logger.debug("Config saved to %s", CONFIG_FILE)


def get(key: str, default: Any = None) -> Any:
    """Convenience: load config and return a single key."""
    cfg = load_config()
    return cfg.get(key, default)


def set_value(key: str, value: Any) -> None:
    """Convenience: load config, update one key, save."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
