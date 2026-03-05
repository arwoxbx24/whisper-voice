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
from .error_handler import categorize_error, show_error_from_thread

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

        # Recording timer (auto-stop after max_recording_seconds)
        self._recording_timer: Optional[threading.Timer] = None
        self._recording_start_time: float = 0.0

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

            # Start auto-stop timer
            max_sec = self.config.get("max_recording_seconds", 300)
            if max_sec and max_sec > 0:
                self._recording_start_time = time.time()
                self._recording_timer = threading.Timer(max_sec, self._auto_stop_recording)
                self._recording_timer.daemon = True
                self._recording_timer.start()
                # Start countdown display
                if self._ui and self._ui._root:
                    self._ui._root.after(0, self._update_recording_countdown)
        except Exception as exc:
            logger.error("Failed to start recording: %s", exc)
            self._recording = False
            friendly = categorize_error(exc)
            tk_root = self._ui._root if self._ui else None
            show_error_from_thread(
                "Whisper Voice — Ошибка записи",
                friendly,
                tk_root=tk_root,
            )

    def _cancel_recording_timer(self) -> None:
        """Cancel the auto-stop timer (call while holding _lock or before releasing)."""
        if self._recording_timer is not None:
            self._recording_timer.cancel()
            self._recording_timer = None

    def _stop_recording_async(self) -> None:
        """Stop recording and transcribe in a background thread (inside _lock)."""
        if not self._recording:
            return
        self._recording = False
        self._cancel_recording_timer()
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
                friendly = categorize_error(exc)
                tk_root = self._ui._root if self._ui else None
                show_error_from_thread(
                    "Whisper Voice — Ошибка транскрипции",
                    friendly,
                    tk_root=tk_root,
                )
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
    # Auto-stop recording after max duration
    # ------------------------------------------------------------------

    def _auto_stop_recording(self) -> None:
        """Called by threading.Timer when max recording duration is reached."""
        logger.info("Автоматическая остановка записи (достигнут лимит времени)")
        with self._lock:
            if not self._recording:
                return  # already stopped manually
            self._recording_timer = None  # timer already fired — no need to cancel
            self._stop_recording_async()
        # Show notification in UI thread
        if self._ui and self._ui._root:
            self._ui._root.after(0, self._show_auto_stop_notification)

    def _show_auto_stop_notification(self) -> None:
        """Display a brief notification that recording was auto-stopped."""
        try:
            max_sec = self.config.get("max_recording_seconds", 300)
            minutes = max_sec // 60
            msg = f"Запись остановлена автоматически (лимит {minutes} мин)"
            logger.info(msg)
            # Show in UI if it supports notifications; gracefully skip if not
            if self._ui and hasattr(self._ui, "show_notification"):
                self._ui.show_notification(msg)
        except Exception as exc:
            logger.debug("Auto-stop notification error: %s", exc)

    def _update_recording_countdown(self) -> None:
        """Periodic (1 s) callback to update countdown display in UI (tk thread)."""
        if not self._recording:
            # Recording stopped — clear countdown
            if self._ui and hasattr(self._ui, "update_countdown"):
                self._ui.update_countdown(None)
            return
        max_sec = self.config.get("max_recording_seconds", 300)
        if not max_sec or max_sec <= 0:
            return
        elapsed = time.time() - self._recording_start_time
        remaining = max(0, int(max_sec - elapsed))
        if self._ui and hasattr(self._ui, "update_countdown"):
            self._ui.update_countdown(remaining)
        if remaining > 0 and self._ui and self._ui._root:
            self._ui._root.after(1000, self._update_recording_countdown)

    # ------------------------------------------------------------------
    # Cancel recording (from UI "X" button)
    # ------------------------------------------------------------------

    def on_cancel_recording(self) -> None:
        """Called when user cancels recording via the indicator X button."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._cancel_recording_timer()
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
            # Log warning but do NOT raise — main.py already showed a setup dialog.
            # App starts in "no-provider" mode; user can edit config and restart.
            logger.warning(
                "No usable STT provider configured. "
                "Set 'api_key' for OpenAI, 'deepgram_api_key' for Deepgram, "
                "or add 'local' to stt_providers. "
                "Edit ~/.whisper-voice/config.json then restart."
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
            logger.warning(
                "No STT providers could be initialised. "
                "Check API keys and installed packages. "
                "Hotkey will work but transcription will fail until a provider is configured."
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
            if provider_list:
                self._engine = TranscriptionEngine(providers=provider_list)
                logger.info("TranscriptionEngine ready with %d provider(s)", len(provider_list))
            else:
                self._engine = None
                logger.warning("TranscriptionEngine not started — no providers available")
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
            on_settings=self._open_settings,
            on_about=self._open_about,
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

    def _open_settings(self) -> None:
        """Open Setup Wizard as settings dialog (called from tray menu)."""
        import threading
        def _run_wizard():
            try:
                from .setup_wizard import SetupWizard

                def _on_save(new_cfg: dict) -> None:
                    # Update live config and reinit components
                    self.config.update(new_cfg)
                    logger.info("Settings updated via wizard — restarting components")
                    try:
                        self._shutdown()
                        self._init_components(new_cfg)
                        logger.info("Components restarted after settings change")
                    except Exception as exc:
                        logger.error("Failed to restart components after settings: %s", exc)

                wizard = SetupWizard(self.config, on_save=_on_save)
                wizard.run()
            except Exception as exc:
                logger.error("Failed to open settings wizard: %s", exc)

        t = threading.Thread(target=_run_wizard, daemon=True, name="settings-wizard")
        t.start()

    def _open_about(self) -> None:
        """Show 'About' dialog (called from tray menu)."""
        import threading
        def _show():
            try:
                from .setup_wizard import show_about_dialog
                # Pass tk root if available so dialog shows as child window
                root = self._ui._root if self._ui else None
                if root:
                    root.after(0, lambda: show_about_dialog(root))
                else:
                    show_about_dialog()
            except Exception as exc:
                logger.error("Failed to open about dialog: %s", exc)

        t = threading.Thread(target=_show, daemon=True, name="about-dialog")
        t.start()

    @staticmethod
    def _noop() -> None:
        pass
