"""Tests for src/transcriber.py — WhisperTranscriber API client."""
import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _load_config tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        with patch("src.transcriber.CONFIG_PATH", tmp_path / "nonexistent.json"):
            from src.transcriber import _load_config
            result = _load_config()
        assert result == {}

    def test_returns_parsed_json_when_file_exists(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"api_key": "k"}), encoding="utf-8")
        with patch("src.transcriber.CONFIG_PATH", config_file):
            from src.transcriber import _load_config
            result = _load_config()
        assert result["api_key"] == "k"

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("not json", encoding="utf-8")
        with patch("src.transcriber.CONFIG_PATH", config_file):
            from src.transcriber import _load_config
            result = _load_config()
        assert result == {}


# ---------------------------------------------------------------------------
# WhisperTranscriber.__init__
# ---------------------------------------------------------------------------

class TestWhisperTranscriberInit:
    def test_raises_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.CONFIG_PATH", tmp_path / "noconfig.json"):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                WhisperTranscriber()

    def test_uses_explicit_api_key(self):
        from src.transcriber import WhisperTranscriber
        mock_client = MagicMock()
        with patch("src.transcriber.OpenAI", return_value=mock_client) as MockOpenAI:
            t = WhisperTranscriber(api_key="k")
            MockOpenAI.assert_called_once_with(api_key="k")
            assert t.model == "whisper-1"
            assert t.language == "ru"

    def test_uses_env_var_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.CONFIG_PATH", tmp_path / "noconfig.json"), \
             patch("src.transcriber.OpenAI") as MockOpenAI:
            t = WhisperTranscriber()
            MockOpenAI.assert_called_once_with(api_key="k")

    def test_uses_config_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"api_key": "k"}), encoding="utf-8")
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.CONFIG_PATH", config_file), \
             patch("src.transcriber.OpenAI") as MockOpenAI:
            t = WhisperTranscriber()
            MockOpenAI.assert_called_once_with(api_key="k")

    def test_custom_model_and_language(self):
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.OpenAI"):
            t = WhisperTranscriber(api_key="k", model="whisper-2", language="en")
            assert t.model == "whisper-2"
            assert t.language == "en"


# ---------------------------------------------------------------------------
# WhisperTranscriber.transcribe / transcribe_with_prompt
# ---------------------------------------------------------------------------

class TestWhisperTranscriberTranscribe:
    def _make_transcriber(self):
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.OpenAI") as MockOpenAI:
            t = WhisperTranscriber(api_key="k")
            t._client = MockOpenAI.return_value
            return t

    def test_transcribe_raises_file_not_found(self, tmp_path):
        t = self._make_transcriber()
        with pytest.raises(FileNotFoundError):
            t.transcribe(str(tmp_path / "nonexistent.wav"))

    def test_transcribe_returns_string_result(self, tmp_path):
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        t._client.audio.transcriptions.create.return_value = "Привет мир"
        result = t.transcribe(str(audio_file))
        assert result == "Привет мир"

    def test_transcribe_strips_whitespace(self, tmp_path):
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        mock_response = MagicMock()
        mock_response.text = "  Hello world  "
        t._client.audio.transcriptions.create.return_value = mock_response

        result = t.transcribe(str(audio_file))
        assert result == "Hello world"

    def test_transcribe_propagates_openai_error(self, tmp_path):
        from openai import OpenAIError
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        t._client.audio.transcriptions.create.side_effect = OpenAIError("API error")
        with pytest.raises(OpenAIError, match="Whisper API error"):
            t.transcribe(str(audio_file))

    def test_transcribe_with_custom_prompt_passes_prompt(self, tmp_path):
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        t._client.audio.transcriptions.create.return_value = "result"
        t.transcribe_with_prompt(str(audio_file), prompt="custom prompt")

        call_kwargs = t._client.audio.transcriptions.create.call_args[1]
        assert call_kwargs.get("prompt") == "custom prompt"

    def test_transcribe_with_empty_prompt_omits_prompt_key(self, tmp_path):
        """When prompt is empty/falsy, prompt key is omitted from the API call."""
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        t._client.audio.transcriptions.create.return_value = "result"
        t.transcribe_with_prompt(str(audio_file), prompt="")

        call_kwargs = t._client.audio.transcriptions.create.call_args[1]
        # Empty prompt is falsy — the code does `if prompt: kwargs["prompt"] = prompt`
        assert "prompt" not in call_kwargs

    def test_transcribe_uses_default_prompt(self, tmp_path):
        """transcribe() always passes DEFAULT_PROMPT."""
        t = self._make_transcriber()
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        t._client.audio.transcriptions.create.return_value = "result"
        t.transcribe(str(audio_file))

        call_kwargs = t._client.audio.transcriptions.create.call_args[1]
        from src.transcriber import DEFAULT_PROMPT
        assert call_kwargs.get("prompt") == DEFAULT_PROMPT

    def test_transcribe_passes_model_and_language(self, tmp_path):
        from src.transcriber import WhisperTranscriber
        with patch("src.transcriber.OpenAI") as MockOpenAI:
            t = WhisperTranscriber(api_key="k", model="whisper-2", language="en")
            t._client = MockOpenAI.return_value

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)
        t._client.audio.transcriptions.create.return_value = "result"

        t.transcribe(str(audio_file))
        call_kwargs = t._client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-2"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["response_format"] == "text"
