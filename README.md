# Whisper Voice

Lightweight voice-to-text desktop app for Windows 10/11.

## Features
- Push-to-talk and toggle recording modes
- Multi-provider STT: OpenAI Whisper, Deepgram Nova-3, local faster-whisper
- Automatic failover between providers (circuit breaker)
- Audio caching — never lose recordings on network drops
- Smart text insertion with clipboard protection
- System tray with minimal UI

## Download
Go to [Actions](../../actions) -> latest successful build -> download artifacts.

## Configuration
On first run, edit `~/.whisper-voice/config.json` with your API keys.
