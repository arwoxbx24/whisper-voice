"""Tests for src/config.py — load, save, validate, defaults, normalize_hotkey."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# normalize_hotkey
# ---------------------------------------------------------------------------

class TestNormalizeHotkey:
    def test_bare_modifiers_wrapped(self):
        from src.config import normalize_hotkey
        assert normalize_hotkey("ctrl+shift+space") == "<ctrl>+<shift>+space"

    def test_already_wrapped_unchanged(self):
        from src.config import normalize_hotkey
        assert normalize_hotkey("<ctrl>+<shift>+space") == "<ctrl>+<shift>+space"

    def test_partial_wrap(self):
        from src.config import normalize_hotkey
        result = normalize_hotkey("ctrl+<shift>+space")
        assert result == "<ctrl>+<shift>+space"

    def test_single_key_no_modifier(self):
        from src.config import normalize_hotkey
        assert normalize_hotkey("space") == "space"

    def test_alt_modifier(self):
        from src.config import normalize_hotkey
        assert normalize_hotkey("alt+a") == "<alt>+a"

    def test_case_insensitive_modifier(self):
        from src.config import normalize_hotkey
        result = normalize_hotkey("CTRL+SHIFT+space")
        assert result == "<ctrl>+<shift>+space"


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

class TestValidate:
    def setup_method(self):
        from src.config import DEFAULT_CONFIG
        self.valid = dict(DEFAULT_CONFIG)

    def test_valid_config_passes(self):
        from src.config import _validate
        _validate(self.valid)  # No exception

    def test_invalid_hotkey_mode_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["hotkey_mode"] = "invalid_mode"
        with pytest.raises(ConfigError, match="hotkey_mode"):
            _validate(cfg)

    def test_invalid_insert_method_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["insert_method"] = "magic"
        with pytest.raises(ConfigError, match="insert_method"):
            _validate(cfg)

    def test_invalid_stt_provider_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["stt_providers"] = ["unknown_provider"]
        with pytest.raises(ConfigError, match="stt_providers"):
            _validate(cfg)

    def test_stt_providers_not_list_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["stt_providers"] = "openai"
        with pytest.raises(ConfigError, match="stt_providers must be a list"):
            _validate(cfg)

    def test_invalid_log_level_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["log_level"] = "VERBOSE"
        with pytest.raises(ConfigError, match="log_level"):
            _validate(cfg)

    def test_max_recording_seconds_negative_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["max_recording_seconds"] = -1
        with pytest.raises(ConfigError, match="max_recording_seconds"):
            _validate(cfg)

    def test_max_recording_seconds_too_large_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["max_recording_seconds"] = 9999
        with pytest.raises(ConfigError, match="max_recording_seconds"):
            _validate(cfg)

    def test_max_recording_seconds_zero_valid(self):
        from src.config import _validate
        cfg = dict(self.valid)
        cfg["max_recording_seconds"] = 0
        _validate(cfg)  # Should not raise

    def test_api_key_not_string_raises(self):
        from src.config import _validate, ConfigError
        cfg = dict(self.valid)
        cfg["api_key"] = 12345
        with pytest.raises(ConfigError, match="api_key"):
            _validate(cfg)

    def test_all_valid_hotkey_modes(self):
        from src.config import _validate
        from src.config import DEFAULT_CONFIG
        for mode in ("toggle", "hold"):
            cfg = dict(DEFAULT_CONFIG)
            cfg["hotkey_mode"] = mode
            _validate(cfg)  # No exception

    def test_all_valid_insert_methods(self):
        from src.config import _validate
        from src.config import DEFAULT_CONFIG
        for method in ("auto", "clipboard", "xdotool", "type"):
            cfg = dict(DEFAULT_CONFIG)
            cfg["insert_method"] = method
            _validate(cfg)  # No exception


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_creates_defaults_when_missing(self, tmp_path):
        from src import config
        config_file = tmp_path / "config.json"
        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            result = config.load_config()
        from src.config import DEFAULT_CONFIG
        assert result["hotkey_mode"] == DEFAULT_CONFIG["hotkey_mode"]
        assert config_file.exists()

    def test_load_existing_config(self, tmp_path):
        from src import config
        config_file = tmp_path / "config.json"
        user_data = {**__import__("src.config", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG}
        user_data["language"] = "en"
        _write_config(config_file, user_data)

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            result = config.load_config()
        assert result["language"] == "en"

    def test_load_corrupted_json_returns_defaults(self, tmp_path):
        from src import config
        config_file = tmp_path / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("not valid json", encoding="utf-8")

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            result = config.load_config()

        from src.config import DEFAULT_CONFIG
        assert result["language"] == DEFAULT_CONFIG["language"]

    def test_load_merges_defaults_with_user_values(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"
        # Write only partial config — missing keys should come from defaults
        partial = {"language": "de"}
        _write_config(config_file, partial)

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            result = config.load_config()

        assert result["language"] == "de"
        assert result["hotkey_mode"] == DEFAULT_CONFIG["hotkey_mode"]

    def test_load_normalizes_hotkey(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"
        data = dict(DEFAULT_CONFIG)
        data["hotkey"] = "ctrl+shift+space"  # bare modifiers
        _write_config(config_file, data)

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            result = config.load_config()

        assert result["hotkey"] == "<ctrl>+<shift>+space"


class TestSaveConfig:
    def test_save_creates_file(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            config.save_config(dict(DEFAULT_CONFIG))

        assert config_file.exists()
        with open(config_file) as f:
            data = json.load(f)
        assert data["language"] == DEFAULT_CONFIG["language"]

    def test_save_invalid_config_raises(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG, ConfigError
        config_file = tmp_path / "config.json"
        bad_cfg = dict(DEFAULT_CONFIG)
        bad_cfg["hotkey_mode"] = "bad"

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            with pytest.raises(ConfigError):
                config.save_config(bad_cfg)


class TestGetAndSetValue:
    def test_get_returns_value(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            val = config.get("language")
        assert val == DEFAULT_CONFIG["language"]

    def test_get_returns_default_for_missing_key(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            val = config.get("nonexistent_key", "fallback")
        assert val == "fallback"

    def test_set_value_updates_key(self, tmp_path):
        from src import config
        from src.config import DEFAULT_CONFIG
        config_file = tmp_path / "config.json"

        with patch.object(config, "CONFIG_DIR", tmp_path), \
             patch.object(config, "CONFIG_FILE", config_file):
            config.set_value("language", "fr")
            val = config.get("language")
        assert val == "fr"
