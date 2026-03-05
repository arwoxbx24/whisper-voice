"""
Main application class for Whisper Voice.

Ties together:
  - AudioRecorder   — captures microphone input
  - WhisperTranscriber — sends WAV to OpenAI Whisper
  - HotkeyManager   — global hotkey listener
  - TextInserter    — pastes transcribed text
  - UIController    — system-tray icon + recording indicator

Recording flow (toggle mode):
  1. User presses hotkey -> start_recording()
  2. Recording indicator shown
  3. User presses hotkey again -> stop_and_transcribe()
  4. WAV sent to Whisper API -> text returned
  5. Text inserted at cursor
  6. Indicator hidden

Recording flow (hold mode):
  1. User holds hotkey  -> start_recording()
  2. User releases hotkey -> stop_and_transcribe()
"""

import logging
import os
import threading
import time
from typing import Any, List, Optional

from . import config as cfg_module
from .audio_recorder import AudioRecorder
from .transcriber import WhisperTranscriber
from .text_inserter import SmartTextInserter
from .hotkey_manager import HotkeyManager
from .ui import UIController
from .state_machine import StateMachine, State
from .audio_cache import AudioCache
from .network_monitor import NetworkMonitor
from .transcription_engine import TranscriptionEngine
from .providers.openai_provider import OpenAIProvider
from .providers.deepgram_provider import DeepgramProvider
from .providers.local_provider import LocalProvider
from .providers.base import STTProvider

# Backward compat alias
TextInserter = SmartTextInserter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WhisperVoiceApp — main orchestrator
# ---------------------------------------------------------------------------

class WhisperVoiceApp:
    """Main application. Ties all modules together."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._recording = False
        self._processing = False
        self._lock = threading.Lock()
        self._running = False

        # Sub-components (initialised in _init_components())
        self._recorder: Optional[AudioRecorder] = None
        self._transcriber: Optional[WhisperTranscriber] = None
        self._inserter: Optional[SmartTextInserter] = None
        self._hotkey: Optional[HotkeyManager] = None
        self._ui: Optional[UIController] = None

        # New Phase 5 components
        self._state_machine: Optional[StateMachine] = None
        self._audio_cache: Optional[AudioCache] = None
        self._network_monitor: Optional[NetworkMonitor] = None
        self._engine: Optional[TranscriptionEngine] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialise components and block until quit (via UIController.run())."""
        logger.info("Starting Whisper Voice")
        self._validate_config()
        self._init_components()
        self._running = True

        logger.info(
            "Ready. Hotkey=%s mode=%s insert=%s",
            self.config["hotkey"],
            self.config["hotkey_mode"],
            self.config["insert_method"],
        )

        # UIController.run() blocks the main thread (tkinter mainloop)
        # The tray icon and hotkey listeners run in daemon threads.
        try:
            self._ui.run()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        finally:
            self._shutdown()

    def quit(self) -> None:
        """Signal the app to stop (called from UI quit callback)."""
        self._running = False
        if self._ui:
            self._ui.quit()

    # ------------------------------------------------------------------
    # Recording flow
    # ------------------------------------------------------------------

    def on_hotkey_activate(self) -> None:
        """Called when hotkey is pressed (toggle mode: start; hold mode: start)."""
        with self._lock:
            if not self._recording and not self._processing:
                self._start_recording()

    def on_hotkey_deactivate(self) -> None:
        """Called when hotkey is released (hold mode: stop; toggle mode: second press)."""
        with self._lock:
            if self._recording:
                self._stop_recording_async()

    def _start_recording(self) -> None:
        """Start microphone recording (called inside _lock)."""
        try:
            level_cb = self._ui.update_audio_level if self._ui else None
            self._recorder = AudioRecorder(
                sample_rate=16000,
                channels=1,
                dtype="int16",
                level_callback=level_cb,
            )
            self._recorder.start()
            self._recording = True
            logger.info("Recording started")
            if self._ui:
                self._ui.show_recording()
        except Exception as exc:
            logger.error("Failed to start recording: %s", exc)
            self._recording = False

    def _stop_recording_async(self) -> None:
        """Stop recording and transcribe in a background thread (inside _lock)."""
        if not self._recording:
            return
        self._recording = False
        recorder = self._recorder
        self._recorder = None

        self._processing = True

        if self._ui:
            self._ui.hide_recording()

        threading.Thread(
            target=self._stop_and_transcribe,
            args=(recorder,),
            daemon=True,
        ).start()

    def _stop_and_transcribe(self, recorder: AudioRecorder) -> None:
        """Background: stop recorder -> transcribe -> insert text."""
        wav_path = None
        try:
            # 1. Stop recording, get WAV path
            try:
                wav_path = recorder.stop()
                logger.info("Recording saved: %s", wav_path)
            except Exception as exc:
                logger.error("Failed to stop recorder: %s", exc)
                return

            # 2. Transcribe (use engine if available, fallback to legacy transcriber)
            try:
                if self._engine:
                    result = self._engine.transcribe(
                        wav_path,
                        language=self.config.get("language", "ru"),
                    )
                    text = result.text.strip() if result and result.text else ""
                else:
                    text = self._transcriber.transcribe(wav_path)
            except Exception as exc:
                logger.error("Transcription failed: %s", exc)
                # Cache audio if cache is available so it can be retried later
                if self._audio_cache and wav_path:
                    try:
                        self._audio_cache.enqueue(
                            wav_path,
                            language=self.config.get("language", "ru"),
                            prompt=self.config.get("prompt_context", ""),
                        )
                        wav_path = None  # don't delete — cache owns it now
                        logger.info("Audio queued in cache for retry")
                    except Exception as cache_exc:
                        logger.debug("Audio cache enqueue failed: %s", cache_exc)
                return

            if not text:
                logger.info("Empty transcription, nothing to insert")
                return

            # 3. Insert text
            try:
                self._inserter.insert(text)
                logger.info("Text inserted (%d chars)", len(text))
            except Exception as exc:
                logger.error("Text insertion failed: %s", exc)

        finally:
            self._processing = False
            # Cleanup temp WAV
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Cancel recording (from UI "X" button)
    # ------------------------------------------------------------------

    def on_cancel_recording(self) -> None:
        """Called when user cancels recording via the indicator X button."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            recorder = self._recorder
            self._recorder = None

        if recorder:
            # Stop recorder and discard audio without transcribing
            threading.Thread(
                target=self._cancel_recorder,
                args=(recorder,),
                daemon=True,
            ).start()

    def _cancel_recorder(self, recorder: AudioRecorder) -> None:
        """Stop the recorder and throw away the audio file."""
        try:
            wav_path = recorder.stop()
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            logger.info("Recording cancelled")
        except Exception as exc:
            logger.debug("Error cancelling recorder: %s", exc)

    # ------------------------------------------------------------------
    # Init / shutdown helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        cfg = self.config
        providers = cfg.get("stt_providers", ["openai"])
        needs_openai_key = "openai" in providers
        needs_deepgram_key = "deepgram" in providers
        has_local = "local" in providers

        openai_key = cfg.get("api_key", "").strip()
        deepgram_key = cfg.get("deepgram_api_key", "").strip()

        # At least one provider must be usable
        can_openai = needs_openai_key and bool(openai_key)
        can_deepgram = needs_deepgram_key and bool(deepgram_key)
        can_local = has_local  # local doesn't need key (checked at import time)

        if not (can_openai or can_deepgram or can_local):
            raise RuntimeError(
                "No usable STT provider. "
                "Set 'api_key' for OpenAI, 'deepgram_api_key' for Deepgram, "
                "or add 'local' to stt_providers (requires faster-whisper). "
                "Edit ~/.whisper-voice/config.json."
            )

    def _build_providers(self, cfg: dict[str, Any]) -> List[STTProvider]:
        """Build ordered list of STT provider instances from config."""
        providers: List[STTProvider] = []
        for name in cfg.get("stt_providers", ["openai"]):
            try:
                if name == "openai":
                    prov = OpenAIProvider(api_key=cfg.get("api_key", ""))
                elif name == "deepgram":
                    prov = DeepgramProvider(api_key=cfg.get("deepgram_api_key", ""))
                elif name == "local":
                    prov = LocalProvider(
                        model_size=cfg.get("local_whisper_model", "base"),
                        device=cfg.get("local_whisper_device", "cpu"),
                    )
                else:
                    logger.warning("Unknown STT provider %r, skipping", name)
                    continue

                if not prov.is_available():
                    logger.warning("Provider %r not available (missing key/package), skipping", name)
                    continue

                providers.append(prov)
                logger.info("Registered STT provider: %s", name)

            except Exception as exc:
                logger.warning("Failed to initialise provider %r: %s", name, exc)

        if not providers:
            raise RuntimeError(
                "No STT providers could be initialised. "
                "Check API keys and installed packages."
            )
        return providers

    def _wire_state_machine(self) -> None:
        """Register state transition callbacks on the StateMachine."""
        if not self._state_machine:
            return
        sm = self._state_machine

        def on_recording() -> None:
            logger.debug("State -> RECORDING")
            if self._ui:
                self._ui.show_recording()

        def on_processing() -> None:
            logger.debug("State -> PROCESSING")
            if self._ui:
                self._ui.hide_recording()

        def on_idle() -> None:
            logger.debug("State -> IDLE")

        sm.on_enter(State.RECORDING, on_recording)
        sm.on_enter(State.PROCESSING, on_processing)
        sm.on_enter(State.IDLE, on_idle)

    def _on_network_restored(self) -> None:
        """Called by NetworkMonitor when connectivity is restored."""
        logger.info("Network restored — triggering audio cache queue processing")
        if self._audio_cache:
            # Process any queued audio items
            threading.Thread(
                target=self._process_cache_queue,
                daemon=True,
            ).start()

    def _process_cache_queue(self) -> None:
        """Process pending items in the audio cache (called after network restore)."""
        if not self._audio_cache or not self._engine:
            return
        try:
            item = self._audio_cache.get_next_pending()
            while item is not None:
                item_id = item["id"]
                audio_path = item["audio_path"]
                language = item.get("language", "ru")
                try:
                    result = self._engine.transcribe(audio_path, language=language)
                    text = result.text.strip() if result and result.text else ""
                    if text and self._inserter:
                        self._inserter.insert(text)
                        logger.info("Cache queue: inserted %d chars from %s", len(text), audio_path)
                    self._audio_cache.mark_complete(item_id)
                except Exception as exc:
                    logger.error("Cache queue: failed to process item %d: %s", item_id, exc)
                    self._audio_cache.mark_pending(item_id, str(exc))
                item = self._audio_cache.get_next_pending()
        except Exception as exc:
            logger.error("Cache queue processing error: %s", exc)

    def _init_components(self, cfg: Optional[dict[str, Any]] = None) -> None:
        if cfg is None:
            cfg = self.config

        # State machine
        self._state_machine = StateMachine()
        self._wire_state_machine()

        # Audio cache (optional)
        if cfg.get("audio_cache_enabled", True):
            try:
                self._audio_cache = AudioCache()
                logger.info("AudioCache initialised")
            except Exception as exc:
                logger.warning("AudioCache init failed: %s", exc)
                self._audio_cache = None

        # Network monitor
        try:
            self._network_monitor = NetworkMonitor(
                on_connected=self._on_network_restored,
            )
            self._network_monitor.start()
            logger.info("NetworkMonitor started")
        except Exception as exc:
            logger.warning("NetworkMonitor init failed: %s", exc)
            self._network_monitor = None

        # Transcription engine (multi-provider)
        try:
            provider_list = self._build_providers(cfg)
            self._engine = TranscriptionEngine(providers=provider_list)
            logger.info("TranscriptionEngine ready with %d provider(s)", len(provider_list))
        except Exception as exc:
            logger.warning("TranscriptionEngine init failed (%s), falling back to WhisperTranscriber", exc)
            self._engine = None

        # Legacy transcriber fallback (keeps existing flow working)
        self._transcriber = WhisperTranscriber(
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "whisper-1"),
            language=cfg.get("language", "ru"),
        )

        # Smart text inserter
        self._inserter = SmartTextInserter(method=cfg.get("insert_method", "auto"))

        # HotkeyManager
        mode = cfg["hotkey_mode"]
        self._hotkey = HotkeyManager(
            hotkey=cfg["hotkey"],
            mode=mode,
        )
        self._hotkey.set_callback(
            on_activate=self.on_hotkey_activate,
            on_deactivate=self.on_hotkey_deactivate,
        )
        self._hotkey.start()

        # UIController — provides tray icon + floating recording indicator
        self._ui = UIController(
            on_stop_recording=self.on_hotkey_deactivate,
            on_cancel_recording=self.on_cancel_recording,
            on_settings=self._noop,
            on_about=self._noop,
            on_quit=self.quit,
            on_toggle_recording=self.on_hotkey_activate,
        )

    def shutdown(self) -> None:
        """Public graceful shutdown — stops all components."""
        self._shutdown()

    def _shutdown(self) -> None:
        logger.info("Shutting down")
        if self._hotkey:
            try:
                self._hotkey.stop()
            except Exception as exc:
                logger.debug("Hotkey stop error: %s", exc)

        if self._network_monitor:
            try:
                self._network_monitor.stop()
            except Exception as exc:
                logger.debug("NetworkMonitor stop error: %s", exc)

        # Stop any in-progress recording
        with self._lock:
            if self._recorder and self._recording:
                try:
                    self._recorder.stop()
                except Exception:
                    pass
                self._recording = False

    @staticmethod
    def _noop() -> None:
        pass
