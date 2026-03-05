# Whisper Voice

Lightweight voice-to-text desktop application for Windows 10/11, powered by OpenAI Whisper API.
Press a hotkey, speak, and your words appear as text in any application.

## Features

- **Push-to-talk and toggle recording modes** — choose between hold-to-record or click-to-toggle
- **Multi-provider STT** — OpenAI Whisper, Deepgram Nova-3, and local faster-whisper (offline)
- **Automatic failover** — circuit breaker switches providers on API errors
- **Audio caching** — recordings are never lost on network drops (retried automatically)
- **Smart text insertion** — clipboard-based paste with protection of existing clipboard content
- **System tray icon** — minimal footprint, right-click for settings
- **Setup Wizard** — guided first-run configuration with API key test

## Quick Start (Windows)

1. Download `WhisperVoice.exe` from [Actions](../../actions) — latest successful build → download artifacts
2. Run `WhisperVoice.exe`
3. Follow the Setup Wizard: enter your OpenAI API key, choose a hotkey
4. Press `Ctrl+Shift+Space` to start recording
5. Speak, then press again to stop — text appears in the active window

## Requirements

- Windows 10 / 11
- Microphone
- Internet connection (for OpenAI and Deepgram providers)
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Configuration

Config file is stored at `%USERPROFILE%\.whisper-voice\config.json`.

Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey` | `<ctrl>+<shift>+space` | Global recording hotkey |
| `hotkey_mode` | `toggle` | `toggle` or `hold` |
| `stt_providers` | `["openai"]` | Provider priority list |
| `language` | `ru` | Transcription language |
| `audio_cache_enabled` | `true` | Cache audio on network failure |

## STT Providers

| Provider | Cost | Quality | Offline |
|----------|------|---------|---------|
| `openai` | ~$0.006/min | Excellent | No |
| `deepgram` | ~$0.0043/min | Excellent | No |
| `local` | Free | Good | Yes |

To use multiple providers with automatic failover:
```json
"stt_providers": ["openai", "deepgram", "local"]
```

## Building from Source

```bash
pip install -r requirements.txt
python build.py
```

## Running Tests

```bash
python -m pytest tests/ -v --cov=src
```

## Logs

`%USERPROFILE%\.whisper-voice\whisper-voice.log`

## License

MIT
