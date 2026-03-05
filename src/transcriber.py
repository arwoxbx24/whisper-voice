"""
WhisperTranscriber — OpenAI Whisper API client for Russian speech transcription.

Loads API key from ~/.whisper-voice/config.json or env var OPENAI_API_KEY.
Supports mixed Russian/English speech via prompt parameter.
"""

import os
import json
from pathlib import Path
from openai import OpenAI, OpenAIError

# Common tech terms to include in default prompt.
# Helps Whisper recognise English words embedded in Russian speech.
DEFAULT_PROMPT = (
    "Python, JavaScript, TypeScript, Docker, Kubernetes, API, REST, JSON, YAML, "
    "GitHub, GitLab, Claude, Anthropic, OpenAI, GPT, Whisper, ChatGPT, "
    "React, Vue, Node.js, FastAPI, Django, Flask, PostgreSQL, MySQL, Redis, "
    "MongoDB, Nginx, Linux, Ubuntu, Debian, AWS, GCP, Azure, S3, MinIO, "
    "Telegram, WebSocket, HTTP, HTTPS, SSH, SSL, TLS, JWT, OAuth, CI/CD, "
    "PM2, Bash, Zsh, pip, npm, yarn, git, pull request, merge, commit, branch, "
    "IDE, VS Code, PyCharm, IntelliJ, Figma, Notion, Slack, Jira, Bitrix"
)

CONFIG_PATH = Path.home() / ".whisper-voice" / "config.json"


def _load_config() -> dict:
    """Load config from ~/.whisper-voice/config.json, return empty dict if missing."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


class WhisperTranscriber:
    """
    Transcribes audio files using OpenAI Whisper API (cloud).

    API key resolution order:
    1. api_key argument
    2. ~/.whisper-voice/config.json → "openai_api_key"
    3. OPENAI_API_KEY environment variable

    Args:
        api_key: OpenAI API key. If None, loaded from config or env.
        model: Whisper model name. Default: "whisper-1".
        language: BCP-47 language code. Default: "ru" (Russian).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        language: str = "ru",
    ):
        self.model = model
        self.language = language

        # Resolve API key
        resolved_key = api_key
        if not resolved_key:
            config = _load_config()
            # Support both "api_key" (config.py default) and legacy "openai_api_key"
            resolved_key = config.get("api_key") or config.get("openai_api_key")
        if not resolved_key:
            resolved_key = os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set it in:\n"
                f"  {CONFIG_PATH} → {{\"openai_api_key\": \"sk-...\"}}\n"
                "  or env var OPENAI_API_KEY"
            )

        self._client = OpenAI(api_key=resolved_key)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file using the default tech-terms prompt.

        Args:
            audio_path: Absolute or relative path to audio file.
                        Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.

        Returns:
            Transcribed text string.

        Raises:
            FileNotFoundError: If audio file does not exist.
            OpenAIError: On API / network errors (propagated with context).
        """
        return self.transcribe_with_prompt(audio_path, prompt=DEFAULT_PROMPT)

    def transcribe_with_prompt(self, audio_path: str, prompt: str = "") -> str:
        """
        Transcribe audio with a custom context prompt.

        The prompt helps Whisper recognise domain-specific terms, brand names,
        and mixed-language speech. It is NOT a system instruction — Whisper uses
        it as prior context (preceding transcript).

        Args:
            audio_path: Path to audio file.
            prompt: Context string with keywords/brand names. Up to ~224 tokens.

        Returns:
            Transcribed text string.

        Raises:
            FileNotFoundError: If audio file does not exist.
            OpenAIError: On API / network errors.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            with open(path, "rb") as audio_file:
                kwargs: dict = {
                    "model": self.model,
                    "file": audio_file,
                    "language": self.language,
                    "response_format": "text",
                }
                if prompt:
                    kwargs["prompt"] = prompt

                result = self._client.audio.transcriptions.create(**kwargs)

            # response_format="text" returns a plain string
            if isinstance(result, str):
                return result.strip()
            # Fallback: object with .text attribute
            return result.text.strip()

        except OpenAIError as exc:
            raise OpenAIError(
                f"Whisper API error while transcribing '{audio_path}': {exc}"
            ) from exc
