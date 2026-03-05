"""Tests for src/providers/ — OpenAI, Deepgram, Local providers."""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestOpenAIProvider:
    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="")
        assert not p.is_available()

    def test_available_with_key(self):
        from src.providers.openai_provider import OpenAIProvider
        with patch("openai.OpenAI"):
            p = OpenAIProvider(api_key="k")
            assert p.is_available()

    def test_name_is_openai(self):
        from src.providers.openai_provider import OpenAIProvider
        with patch("openai.OpenAI"):
            p = OpenAIProvider(api_key="k")
            assert p.name == "openai"

    def test_transcribe_raises_permanent_when_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import PermanentError
        # Provider with no key — is_available() returns False
        p = OpenAIProvider(api_key="")
        assert not p.is_available()
        # Even if file existed, provider should raise PermanentError
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)
        with pytest.raises(PermanentError):
            p.transcribe(str(audio_file))

    def test_transcribe_rate_limit_raises_transient(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import TransientError
        import openai

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            p._client.audio.transcriptions.create.side_effect = openai.RateLimitError(
                "rate limit", response=MagicMock(status_code=429), body={}
            )
            with pytest.raises(TransientError, match="rate limit"):
                p.transcribe(str(audio_file))

    def test_transcribe_auth_error_raises_permanent(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import PermanentError
        import openai

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            p._client.audio.transcriptions.create.side_effect = openai.AuthenticationError(
                "auth failed", response=MagicMock(status_code=401), body={}
            )
            with pytest.raises(PermanentError, match="auth failed"):
                p.transcribe(str(audio_file))

    def test_transcribe_success_returns_result(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import TranscriptionResult

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            mock_resp = MagicMock()
            mock_resp.text = "  Привет мир  "
            p._client.audio.transcriptions.create.return_value = mock_resp

            result = p.transcribe(str(audio_file))
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Привет мир"
            assert result.provider == "openai"
            assert result.duration_ms >= 0

    def test_transcribe_connection_error_raises_transient(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import TransientError
        import openai

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            p._client.audio.transcriptions.create.side_effect = openai.APIConnectionError(
                request=MagicMock()
            )
            with pytest.raises(TransientError, match="connection error"):
                p.transcribe(str(audio_file))

    def test_transcribe_api_status_401_raises_permanent(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import PermanentError
        import openai

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            p._client.audio.transcriptions.create.side_effect = openai.APIStatusError(
                "unauthorized", response=mock_resp, body={}
            )
            with pytest.raises(PermanentError):
                p.transcribe(str(audio_file))

    def test_transcribe_api_status_500_raises_transient(self, tmp_path):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import TransientError
        import openai

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 36)

        with patch("openai.OpenAI") as MockOpenAI:
            p = OpenAIProvider(api_key="k")
            p._client = MockOpenAI.return_value
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            p._client.audio.transcriptions.create.side_effect = openai.APIStatusError(
                "server error", response=mock_resp, body={}
            )
            with pytest.raises(TransientError):
                p.transcribe(str(audio_file))

    def test_transcribe_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        from src.providers.openai_provider import OpenAIProvider
        with patch("openai.OpenAI"):
            p = OpenAIProvider()
            assert p._api_key == "k"


class TestDeepgramProvider:
    def test_name_is_deepgram(self):
        from src.providers.deepgram_provider import DeepgramProvider
        p = DeepgramProvider(api_key="k")
        assert p.name == "deepgram"

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        from src.providers.deepgram_provider import DeepgramProvider
        p = DeepgramProvider(api_key="")
        assert not p.is_available()

    def test_available_with_key(self):
        from src.providers.deepgram_provider import DeepgramProvider
        with patch("deepgram.DeepgramClient"):
            p = DeepgramProvider(api_key="k")
            assert p.is_available()


class TestLocalProvider:
    def test_name_is_local(self):
        from src.providers.local_provider import LocalProvider
        p = LocalProvider()
        assert p.name == "local"

    def test_not_available_without_faster_whisper(self):
        from src.providers.local_provider import LocalProvider
        import sys
        original = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = None
        try:
            p = LocalProvider()
            assert not p.is_available()
        finally:
            if original is None:
                sys.modules.pop("faster_whisper", None)
            else:
                sys.modules["faster_whisper"] = original

    def test_not_available_when_model_not_loaded(self):
        from src.providers.local_provider import LocalProvider
        p = LocalProvider()
        # _model is None by default without loading
        assert p._model is None
        assert not p.is_available()


class TestBaseProvider:
    def test_transcription_result_fields(self):
        from src.providers.base import TranscriptionResult
        r = TranscriptionResult(text="hello", provider="test", duration_ms=123.4)
        assert r.text == "hello"
        assert r.provider == "test"
        assert r.duration_ms == 123.4

    def test_transient_error_is_exception(self):
        from src.providers.base import TransientError
        with pytest.raises(TransientError):
            raise TransientError("transient")

    def test_permanent_error_is_exception(self):
        from src.providers.base import PermanentError
        with pytest.raises(PermanentError):
            raise PermanentError("permanent")

    def test_transcription_result_default_duration(self):
        from src.providers.base import TranscriptionResult
        r = TranscriptionResult(text="x", provider="p", duration_ms=0)
        assert r.duration_ms == 0
