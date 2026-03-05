"""
Setup Wizard for Whisper Voice.

Opens automatically on first launch (no config.json) or when user
requests it via tray menu → "Настройки".

Steps:
  1. Добро пожаловать  — описание приложения
  2. Провайдер STT     — выбор OpenAI / Deepgram / Local
  3. API ключ          — ввод ключа (динамически по выбранному провайдеру)
  4. Горячая клавиша   — выбор комбинации
  5. Готово!           — кнопка запуска

Usage:
    from src.setup_wizard import SetupWizard
    wizard = SetupWizard(config, on_save=lambda cfg: ...)
    wizard.run()   # blocks until wizard is closed
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VERSION — shown on the "About" dialog and wizard header
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
_BG = "#1a1a2e"          # dark navy background
_BG2 = "#16213e"         # slightly lighter panel
_ACCENT = "#0f3460"      # accent blue
_HIGHLIGHT = "#e94560"   # highlight / active color
_TEXT = "#eaeaea"        # main text
_TEXT_DIM = "#8888aa"    # dimmed / placeholder text
_BORDER = "#2a2a4a"      # border color
_GREEN = "#4caf50"
_RED = "#e53935"
_YELLOW = "#ffc107"
_FONT = ("Segoe UI", 10)
_FONT_BIG = ("Segoe UI", 13, "bold")
_FONT_SMALL = ("Segoe UI", 9)


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------

class SetupWizard:
    """Multi-step setup wizard built with tkinter."""

    NUM_STEPS = 5

    def __init__(
        self,
        config: Dict[str, Any],
        on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_start_app: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            config:       Current config dict (will be deep-copied inside wizard).
            on_save:      Called with the updated config when user clicks Save/Finish.
            on_start_app: Called after wizard completes (optional, to start main app).
        """
        import copy
        self._config = copy.deepcopy(config)
        self._on_save = on_save
        self._on_start_app = on_start_app
        self._current_step = 0
        self._root = None
        self._step_frames: list = []
        self._status_var = None
        self._key_entry = None
        self._hotkey_display_var = None
        self._mode_var = None
        self._show_key_var = None
        self._recording_hotkey = False
        self._pressed_keys: set = set()
        self._captured_hotkey: str = ""
        self._summary_var = None
        self._autostart_var = None
        self._api_status_var = None
        self._api_status_lbl = None
        self._test_btn = None
        self._capture_btn = None
        self._capture_status_var = None
        self._btn_back = None
        self._btn_next = None
        self._step_label = None
        self._prog_bar = None
        self._tk = None
        # Provider step state
        self._provider_var = None
        self._openai_frame = None
        self._deepgram_frame = None
        self._local_frame = None
        self._deepgram_key_entry = None
        self._deepgram_status_var = None
        self._deepgram_status_lbl = None
        self._deepgram_test_btn = None
        self._key_var = None
        self._dg_key_var = None
        self._key_check_timer = None
        self._dg_key_check_timer = None
        self._local_model_var = None
        self._local_status_var = None
        self._local_download_btn = None
        self._local_progress_var = None
        self._local_progress_bar = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the wizard window and block until it is closed."""
        try:
            import tkinter as tk
            self._tk = tk
            self._build(tk)
            self._root.mainloop()
        except Exception as exc:
            log.error("SetupWizard crashed: %s", exc, exc_info=True)

    def show_in_thread(self) -> None:
        """Open wizard non-blocking (runs in daemon thread)."""
        t = threading.Thread(target=self.run, daemon=True, name="setup-wizard")
        t.start()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, tk) -> None:
        root = tk.Tk()
        root.title("Whisper Voice — Настройка")
        root.configure(bg=_BG)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Center window
        win_w, win_h = 540, 500
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - win_w) // 2
        y = (sh - win_h) // 2
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        self._root = root

        # ---- Header strip ----
        header = tk.Frame(root, bg=_ACCENT, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Whisper Voice",
            bg=_ACCENT,
            fg=_TEXT,
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).pack(side="left", padx=18, pady=12)

        self._step_label = tk.Label(
            header,
            text="",
            bg=_ACCENT,
            fg=_TEXT_DIM,
            font=_FONT_SMALL,
        )
        self._step_label.pack(side="right", padx=18)

        # ---- Progress bar ----
        prog_outer = tk.Frame(root, bg=_BORDER, height=4)
        prog_outer.pack(fill="x")
        prog_outer.pack_propagate(False)

        self._prog_bar = tk.Frame(prog_outer, bg=_HIGHLIGHT, height=4)
        self._prog_bar.place(relx=0, rely=0, relwidth=0.25, relheight=1)

        # ---- Content area ----
        content = tk.Frame(root, bg=_BG, padx=28, pady=16)
        content.pack(fill="both", expand=True)
        self._content = content

        # ---- Bottom nav buttons ----
        btn_frame = tk.Frame(root, bg=_BG, padx=28, pady=10)
        btn_frame.pack(fill="x", side="bottom")

        self._btn_back = self._make_btn(btn_frame, "\u2190 Назад", self._go_back, secondary=True)
        self._btn_back.pack(side="left")

        self._btn_next = self._make_btn(btn_frame, "Далее \u2192", self._go_next)
        self._btn_next.pack(side="right")

        # ---- Status bar ----
        status_frame = tk.Frame(root, bg=_BG2, height=28)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)

        self._status_var = tk.StringVar(value="")
        tk.Label(
            status_frame,
            textvariable=self._status_var,
            bg=_BG2,
            fg=_TEXT_DIM,
            font=_FONT_SMALL,
            anchor="w",
        ).pack(side="left", padx=12, pady=4)

        # ---- Build step frames ----
        self._step_frames = [
            self._build_step0(content),
            self._build_step1(content),
            self._build_step2(content),
            self._build_step3(content),
            self._build_step4(content),
        ]

        self._show_step(0)

    # ------------------------------------------------------------------
    # Step builders
    # ------------------------------------------------------------------

    def _build_step0(self, parent) -> object:
        """Step 1 — Welcome."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Добро пожаловать!",
            bg=_BG, fg=_TEXT,
            font=_FONT_BIG,
            anchor="w",
        ).pack(anchor="w", pady=(0, 12))

        intro = (
            "Whisper Voice — приложение для голосового ввода текста.\n\n"
            "Как это работает:\n"
            "  1. Нажмите горячую клавишу — запись начнётся\n"
            "  2. Говорите — приложение слушает микрофон\n"
            "  3. Нажмите снова (или отпустите) — текст\n"
            "     появится там, где находится курсор\n\n"
            "Распознавание речи выполняется через OpenAI Whisper —\n"
            "облачный сервис с высокой точностью для русского языка.\n\n"
            "Этот мастер поможет настроить приложение за несколько шагов."
        )

        txt = tk.Text(
            frame,
            bg=_BG2, fg=_TEXT,
            font=_FONT,
            relief="flat",
            bd=0,
            wrap="word",
            height=11,
            cursor="arrow",
            highlightthickness=1,
            highlightbackground=_BORDER,
            padx=10, pady=8,
        )
        txt.insert("1.0", intro)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)

        return frame

    def _build_step1(self, parent) -> object:
        """Step 2 — STT Provider Selection."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Выбор провайдера распознавания речи",
            bg=_BG, fg=_TEXT,
            font=_FONT_BIG,
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            frame,
            text="Выберите сервис для распознавания вашей речи:",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        # Determine initial provider from config
        stt_providers = self._config.get("stt_providers", ["openai"])
        if isinstance(stt_providers, list) and stt_providers:
            initial_provider = stt_providers[0]
        else:
            initial_provider = "openai"

        self._provider_var = tk.StringVar(value=initial_provider)

        providers = [
            ("openai",   "OpenAI Whisper     — облако, $0.006/мин, высокая точность"),
            ("deepgram", "Deepgram Nova-3    — облако, $0.0043/мин, быстрее и дешевле"),
            ("local",    "Local faster-whisper — бесплатно, оффлайн, без интернета"),
        ]

        radio_frame = tk.Frame(frame, bg=_BG2,
                               highlightthickness=1, highlightbackground=_BORDER)
        radio_frame.pack(fill="x", ipady=6)

        for val, label in providers:
            tk.Radiobutton(
                radio_frame,
                text=label,
                variable=self._provider_var,
                value=val,
                command=self._on_provider_change,
                bg=_BG2, fg=_TEXT,
                selectcolor=_ACCENT,
                activebackground=_BG2,
                activeforeground=_TEXT,
                font=_FONT,
                anchor="w",
            ).pack(anchor="w", padx=12, pady=4)

        # Description panel
        self._provider_desc_var = tk.StringVar(value="")
        self._provider_desc_lbl = tk.Label(
            frame,
            textvariable=self._provider_desc_var,
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
            wraplength=460,
        )
        self._provider_desc_lbl.pack(anchor="w", pady=(12, 0))

        self._update_provider_desc()

        return frame

    def _on_provider_change(self) -> None:
        """Called when provider radio button changes."""
        self._update_provider_desc()
        if self._provider_var:
            provider = self._provider_var.get()
            self._config["stt_providers"] = [provider]

    def _update_provider_desc(self) -> None:
        """Update the provider description label."""
        if not hasattr(self, "_provider_desc_var") or self._provider_desc_var is None:
            return
        if not self._provider_var:
            return
        provider = self._provider_var.get()
        descs = {
            "openai": (
                "OpenAI Whisper — проверенный облачный сервис. "
                "Требует API-ключ от platform.openai.com. "
                "Отличное качество для русского языка."
            ),
            "deepgram": (
                "Deepgram Nova-3 — быстрый и дешёвый облачный сервис. "
                "Требует API-ключ от console.deepgram.com. "
                "Низкая задержка, хорошее качество."
            ),
            "local": (
                "Local faster-whisper — работает полностью оффлайн. "
                "Не требует интернета и API-ключа. "
                "Требует скачивания модели (39 МБ — 1.5 ГБ)."
            ),
        }
        self._provider_desc_var.set(descs.get(provider, ""))

    def _build_step2(self, parent) -> object:
        """Step 3 — API Key (dynamic by provider)."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Настройка API ключа",
            bg=_BG, fg=_TEXT,
            font=_FONT_BIG,
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))

        self._api_step_subtitle = tk.Label(
            frame,
            text="",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT,
            justify="left",
        )
        self._api_step_subtitle.pack(anchor="w", pady=(0, 10))

        # ---- OpenAI frame ----
        self._openai_frame = tk.Frame(frame, bg=_BG)

        entry_frame_oa = tk.Frame(self._openai_frame, bg=_BG2,
                                  highlightthickness=1, highlightbackground=_BORDER)
        entry_frame_oa.pack(fill="x", ipady=2)

        tk.Label(
            entry_frame_oa,
            text="sk-",
            bg=_BG2, fg=_TEXT_DIM,
            font=_FONT,
        ).pack(side="left", padx=(8, 0))

        self._key_var = tk.StringVar()
        self._key_entry = tk.Entry(
            entry_frame_oa,
            bg=_BG2, fg=_TEXT,
            font=_FONT,
            relief="flat",
            highlightthickness=0,
            show="\u2022",
            insertbackground=_TEXT,
            textvariable=self._key_var,
        )
        self._key_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(2, 8))
        self._key_var.trace_add("write", self._on_key_changed)

        existing_key = self._config.get("api_key", "")
        if existing_key:
            self._key_entry.insert(0, existing_key)

        self._show_key_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            entry_frame_oa,
            text="Показать",
            variable=self._show_key_var,
            command=self._toggle_key_visibility,
            bg=_BG2, fg=_TEXT_DIM,
            selectcolor=_BG2,
            activebackground=_BG2,
            activeforeground=_TEXT,
            font=_FONT_SMALL,
            relief="flat",
            bd=0,
        ).pack(side="right", padx=8)

        btn_row_oa = tk.Frame(self._openai_frame, bg=_BG)
        btn_row_oa.pack(fill="x", pady=(10, 0))

        self._test_btn = self._make_btn(btn_row_oa, "Проверить ключ", self._test_api_key)
        self._test_btn.pack(side="left")

        self._api_status_var = tk.StringVar(value="")
        self._api_status_lbl = tk.Label(
            btn_row_oa,
            textvariable=self._api_status_var,
            bg=_BG,
            fg=_GREEN,
            font=_FONT,
        )
        self._api_status_lbl.pack(side="left", padx=(12, 0))

        tk.Label(
            self._openai_frame,
            text="\nСтоимость: ~$0.006 за минуту (~0.5 руб). Ключ хранится локально.",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        # ---- Deepgram frame ----
        self._deepgram_frame = tk.Frame(frame, bg=_BG)

        entry_frame_dg = tk.Frame(self._deepgram_frame, bg=_BG2,
                                  highlightthickness=1, highlightbackground=_BORDER)
        entry_frame_dg.pack(fill="x", ipady=2)

        self._dg_key_var = tk.StringVar()
        self._deepgram_key_entry = tk.Entry(
            entry_frame_dg,
            bg=_BG2, fg=_TEXT,
            font=_FONT,
            relief="flat",
            highlightthickness=0,
            show="\u2022",
            insertbackground=_TEXT,
            textvariable=self._dg_key_var,
        )
        self._deepgram_key_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=8)
        self._dg_key_var.trace_add("write", self._on_dg_key_changed)

        existing_dg = self._config.get("deepgram_api_key", "")
        if existing_dg:
            self._deepgram_key_entry.insert(0, existing_dg)

        btn_row_dg = tk.Frame(self._deepgram_frame, bg=_BG)
        btn_row_dg.pack(fill="x", pady=(10, 0))

        self._deepgram_test_btn = self._make_btn(
            btn_row_dg, "Проверить ключ", self._test_deepgram_key
        )
        self._deepgram_test_btn.pack(side="left")

        self._deepgram_status_var = tk.StringVar(value="")
        self._deepgram_status_lbl = tk.Label(
            btn_row_dg,
            textvariable=self._deepgram_status_var,
            bg=_BG,
            fg=_GREEN,
            font=_FONT,
        )
        self._deepgram_status_lbl.pack(side="left", padx=(12, 0))

        tk.Label(
            self._deepgram_frame,
            text="\nПолучите ключ на console.deepgram.com\nСтоимость: ~$0.0043 за минуту (~0.4 руб).",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        # ---- Local frame ----
        self._local_frame = tk.Frame(frame, bg=_BG)

        tk.Label(
            self._local_frame,
            text="Выберите размер модели:",
            bg=_BG, fg=_TEXT,
            font=_FONT,
        ).pack(anchor="w", pady=(0, 6))

        model_sizes = [
            ("tiny",   "tiny   — 39 МБ,   быстро / низкая точность"),
            ("base",   "base   — 74 МБ,   рекомендуется"),
            ("small",  "small  — 244 МБ,  хорошая точность"),
            ("medium", "medium — 769 МБ,  высокая точность"),
            ("large",  "large  — 1550 МБ, максимальная точность"),
        ]

        current_model = self._config.get("local_whisper_model", "base")
        self._local_model_var = tk.StringVar(value=current_model)

        model_frame = tk.Frame(self._local_frame, bg=_BG2,
                               highlightthickness=1, highlightbackground=_BORDER)
        model_frame.pack(fill="x", ipady=4)

        for val, label in model_sizes:
            tk.Radiobutton(
                model_frame,
                text=label,
                variable=self._local_model_var,
                value=val,
                bg=_BG2, fg=_TEXT,
                selectcolor=_ACCENT,
                activebackground=_BG2,
                activeforeground=_TEXT,
                font=_FONT_SMALL,
                anchor="w",
            ).pack(anchor="w", padx=10, pady=2)

        local_btn_row = tk.Frame(self._local_frame, bg=_BG)
        local_btn_row.pack(fill="x", pady=(10, 0))

        self._local_download_btn = self._make_btn(
            local_btn_row, "Проверить / Скачать", self._check_local_model
        )
        self._local_download_btn.pack(side="left")

        self._local_status_var = tk.StringVar(value="")
        tk.Label(
            local_btn_row,
            textvariable=self._local_status_var,
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
        ).pack(side="left", padx=(12, 0))

        # Skip hint
        tk.Label(
            frame,
            text="Можно пропустить и ввести ключ позже через Настройки.",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))

        # Show correct provider sub-frame on entering the step
        # (will be called in _show_step)
        return frame

    def _show_api_step_for_provider(self) -> None:
        """Show the correct sub-frame inside the API key step."""
        if not self._provider_var:
            return
        provider = self._provider_var.get()

        # Update subtitle
        subtitles = {
            "openai":   "Введите API ключ OpenAI (platform.openai.com/api-keys):",
            "deepgram": "Введите API ключ Deepgram (console.deepgram.com):",
            "local":    "Выберите размер модели для локального распознавания:",
        }
        if hasattr(self, "_api_step_subtitle") and self._api_step_subtitle:
            try:
                self._api_step_subtitle.config(text=subtitles.get(provider, ""))
            except Exception:
                pass

        # Hide all, show selected
        for frame_attr in ("_openai_frame", "_deepgram_frame", "_local_frame"):
            frm = getattr(self, frame_attr, None)
            if frm:
                try:
                    frm.pack_forget()
                except Exception:
                    pass

        target_map = {
            "openai":   "_openai_frame",
            "deepgram": "_deepgram_frame",
            "local":    "_local_frame",
        }
        target_attr = target_map.get(provider)
        if target_attr:
            frm = getattr(self, target_attr, None)
            if frm:
                try:
                    frm.pack(fill="both", expand=True)
                except Exception:
                    pass

    def _build_step3(self, parent) -> object:
        """Step 4 — Hotkey."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Горячая клавиша",
            bg=_BG, fg=_TEXT,
            font=_FONT_BIG,
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            frame,
            text="Нажмите кнопку ниже, затем зажмите нужную комбинацию клавиш.",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # Hotkey display
        current_hotkey = self._config.get("hotkey", "<ctrl>+<shift>+space")
        self._hotkey_display_var = tk.StringVar(
            value=self._format_hotkey_display(current_hotkey)
        )

        hotkey_frame = tk.Frame(
            frame, bg=_BG2,
            highlightthickness=1, highlightbackground=_BORDER,
        )
        hotkey_frame.pack(fill="x")

        tk.Label(
            hotkey_frame,
            textvariable=self._hotkey_display_var,
            bg=_BG2, fg=_TEXT,
            font=("Segoe UI", 12, "bold"),
            anchor="center",
            pady=10,
        ).pack(fill="x")

        # Buttons row
        btn_row = tk.Frame(frame, bg=_BG)
        btn_row.pack(fill="x", pady=(10, 0))

        self._capture_btn = self._make_btn(
            btn_row, "Задать клавишу", self._start_hotkey_capture
        )
        self._capture_btn.pack(side="left")

        self._make_btn(
            btn_row, "Сбросить", self._reset_hotkey, secondary=True
        ).pack(side="left", padx=(8, 0))

        self._capture_status_var = tk.StringVar(value="")
        tk.Label(
            frame,
            textvariable=self._capture_status_var,
            bg=_BG, fg=_YELLOW,
            font=_FONT_SMALL,
        ).pack(anchor="w", pady=(6, 0))

        # Mode selection
        tk.Label(
            frame,
            text="Режим записи:",
            bg=_BG, fg=_TEXT,
            font=_FONT,
            anchor="w",
        ).pack(anchor="w", pady=(14, 4))

        self._mode_var = tk.StringVar(value=self._config.get("hotkey_mode", "toggle"))

        mode_frame = tk.Frame(frame, bg=_BG)
        mode_frame.pack(anchor="w")

        tk.Radiobutton(
            mode_frame,
            text="Переключение: нажал — запись, нажал снова — стоп",
            variable=self._mode_var,
            value="toggle",
            bg=_BG, fg=_TEXT,
            selectcolor=_BG2,
            activebackground=_BG,
            activeforeground=_TEXT,
            font=_FONT,
        ).pack(anchor="w")

        tk.Radiobutton(
            mode_frame,
            text="Удержание: держишь кнопку — запись, отпустил — стоп",
            variable=self._mode_var,
            value="hold",
            bg=_BG, fg=_TEXT,
            selectcolor=_BG2,
            activebackground=_BG,
            activeforeground=_TEXT,
            font=_FONT,
        ).pack(anchor="w")

        return frame

    def _build_step4(self, parent) -> object:
        """Step 5 — Done."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Всё готово!",
            bg=_BG, fg=_GREEN,
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 12))

        self._summary_var = tk.StringVar(value="")
        tk.Label(
            frame,
            textvariable=self._summary_var,
            bg=_BG, fg=_TEXT,
            font=_FONT,
            justify="left",
        ).pack(anchor="w", pady=(0, 14))

        tips = (
            "Советы по использованию:\n\n"
            "  • Говорите чётко, в нормальном темпе\n"
            "  • Работает в любом приложении (браузер, мессенджеры, редакторы)\n"
            "  • Иконка в трее: правый клик → Настройки / Выход\n"
            "  • Логи: ~/.whisper-voice/whisper-voice.log\n\n"
            "Нажмите «Начать работу» — приложение свернётся в трей."
        )

        txt = tk.Text(
            frame,
            bg=_BG2, fg=_TEXT,
            font=_FONT,
            relief="flat",
            bd=0,
            wrap="word",
            height=8,
            cursor="arrow",
            highlightthickness=1,
            highlightbackground=_BORDER,
            padx=10, pady=8,
        )
        txt.insert("1.0", tips)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)

        # Autostart checkbox
        autostart_frame = tk.Frame(frame, bg=_BG)
        autostart_frame.pack(anchor="w", pady=(10, 0))

        self._autostart_var = tk.BooleanVar(
            value=bool(self._config.get("auto_start", False))
        )
        tk.Checkbutton(
            autostart_frame,
            text="Запускать при входе в систему",
            variable=self._autostart_var,
            bg=_BG, fg=_TEXT,
            selectcolor=_BG2,
            activebackground=_BG,
            activeforeground=_TEXT,
            font=_FONT,
            relief="flat",
            bd=0,
        ).pack(side="left")

        return frame

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _show_step(self, step: int) -> None:
        for i, frame in enumerate(self._step_frames):
            if i == step:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        self._current_step = step
        labels = [
            "Шаг 1 из 5 — Добро пожаловать",
            "Шаг 2 из 5 — Провайдер STT",
            "Шаг 3 из 5 — API ключ",
            "Шаг 4 из 5 — Горячая клавиша",
            "Шаг 5 из 5 — Готово",
        ]
        self._step_label.config(text=labels[step])
        progress = (step + 1) / self.NUM_STEPS
        self._prog_bar.place(relwidth=progress)

        self._btn_back.config(state="normal" if step > 0 else "disabled")

        if step == self.NUM_STEPS - 1:
            self._btn_next.config(text="Начать работу \u2713", command=self._finish)
            self._update_summary()
        else:
            self._btn_next.config(text="Далее \u2192", command=self._go_next)

        # When entering API key step (step 2), show correct provider sub-frame
        if step == 2:
            self._show_api_step_for_provider()

        if self._status_var:
            self._status_var.set("")

    def _go_next(self) -> None:
        if self._current_step == 1:
            # Save provider selection
            if self._provider_var:
                provider = self._provider_var.get()
                self._config["stt_providers"] = [provider]
        elif self._current_step == 2:
            # Save API key(s) depending on provider
            provider = (self._config.get("stt_providers") or ["openai"])[0]
            if provider == "openai":
                key = self._key_entry.get().strip() if self._key_entry else ""
                if key:
                    self._config["api_key"] = key
            elif provider == "deepgram":
                key = self._deepgram_key_entry.get().strip() if self._deepgram_key_entry else ""
                if key:
                    self._config["deepgram_api_key"] = key
            elif provider == "local":
                if self._local_model_var:
                    self._config["local_whisper_model"] = self._local_model_var.get()
        elif self._current_step == 3:
            # Save hotkey settings
            if self._captured_hotkey:
                self._config["hotkey"] = self._captured_hotkey
            if self._mode_var:
                self._config["hotkey_mode"] = self._mode_var.get()

        next_step = self._current_step + 1
        if next_step < self.NUM_STEPS:
            self._show_step(next_step)

    def _go_back(self) -> None:
        prev_step = self._current_step - 1
        if prev_step >= 0:
            self._show_step(prev_step)

    def _finish(self) -> None:
        """Collect all settings, save config, close wizard."""
        # Final provider
        if self._provider_var:
            provider = self._provider_var.get()
            self._config["stt_providers"] = [provider]
        else:
            provider = (self._config.get("stt_providers") or ["openai"])[0]

        # Final API key(s)
        if provider == "openai":
            if self._key_entry:
                key = self._key_entry.get().strip()
                if key:
                    self._config["api_key"] = key
        elif provider == "deepgram":
            if self._deepgram_key_entry:
                key = self._deepgram_key_entry.get().strip()
                if key:
                    self._config["deepgram_api_key"] = key
        elif provider == "local":
            if self._local_model_var:
                self._config["local_whisper_model"] = self._local_model_var.get()

        # Autostart
        if self._autostart_var:
            self._config["auto_start"] = bool(self._autostart_var.get())

        # Final hotkey
        if self._captured_hotkey:
            self._config["hotkey"] = self._captured_hotkey
        if self._mode_var:
            self._config["hotkey_mode"] = self._mode_var.get()

        # Autostart setting
        if self._autostart_var is not None:
            auto_start_enabled = bool(self._autostart_var.get())
            self._config["auto_start"] = auto_start_enabled
            try:
                from . import autostart as autostart_module
                autostart_module.sync_autostart(auto_start_enabled)
                log.info("Autostart from wizard: %s", "enabled" if auto_start_enabled else "disabled")
            except Exception as exc:
                log.warning("Autostart sync in wizard failed (non-fatal): %s", exc)

        # Persist
        try:
            from . import config as cfg_module
            cfg_module.save_config(self._config)
            log.info("Config saved via SetupWizard")
        except Exception as exc:
            log.error("Failed to save config from wizard: %s", exc)
            if self._status_var:
                self._status_var.set(f"Ошибка сохранения: {exc}")
            return

        if self._on_save:
            try:
                self._on_save(self._config)
            except Exception as exc:
                log.error("on_save callback failed: %s", exc)

        self._close_root()

        if self._on_start_app:
            try:
                self._on_start_app()
            except Exception as exc:
                log.error("on_start_app callback failed: %s", exc)

    def _on_close(self) -> None:
        self._close_root()

    def _close_root(self) -> None:
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    # ------------------------------------------------------------------
    # API Key test
    # ------------------------------------------------------------------

    def _on_key_changed(self, *args) -> None:
        """Debounced auto-validation triggered on OpenAI key entry change."""
        if self._key_check_timer:
            self._root.after_cancel(self._key_check_timer)
            self._key_check_timer = None
        key = self._key_var.get().strip() if self._key_var else ""
        if not key:
            return
        if not key.startswith("sk-"):
            if self._api_status_var:
                self._api_status_var.set("Ключ должен начинаться с sk-")
            if self._api_status_lbl:
                self._api_status_lbl.config(fg=_YELLOW)
            return
        if len(key) >= 20:
            if self._api_status_var:
                self._api_status_var.set("Проверяю...")
            if self._api_status_lbl:
                self._api_status_lbl.config(fg=_TEXT_DIM)
            self._key_check_timer = self._root.after(500, self._test_api_key)

    def _on_dg_key_changed(self, *args) -> None:
        """Debounced auto-validation triggered on Deepgram key entry change."""
        if self._dg_key_check_timer:
            self._root.after_cancel(self._dg_key_check_timer)
            self._dg_key_check_timer = None
        key = self._dg_key_var.get().strip() if self._dg_key_var else ""
        if len(key) >= 20:
            if self._deepgram_status_var:
                self._deepgram_status_var.set("Проверяю...")
            if self._deepgram_status_lbl:
                self._deepgram_status_lbl.config(fg=_TEXT_DIM)
            self._dg_key_check_timer = self._root.after(500, self._test_deepgram_key)

    def _test_api_key(self) -> None:
        key = self._key_entry.get().strip() if self._key_entry else ""
        if not key:
            if self._api_status_var:
                self._api_status_var.set("Введите ключ")
            if self._api_status_lbl:
                self._api_status_lbl.config(fg=_YELLOW)
            return

        if self._api_status_var:
            self._api_status_var.set("Проверяю...")
        if self._api_status_lbl:
            self._api_status_lbl.config(fg=_TEXT_DIM)
        if self._test_btn:
            self._test_btn.config(state="disabled")

        def _check():
            ok, msg = _check_openai_key(key)
            if self._root:
                self._root.after(0, lambda: self._show_api_result(ok, msg))

        threading.Thread(target=_check, daemon=True).start()

    def _show_api_result(self, ok: bool, msg: str) -> None:
        prefix = "\u2713 " if ok else "\u2717 "
        if self._api_status_var:
            self._api_status_var.set(prefix + msg)
        if self._api_status_lbl:
            self._api_status_lbl.config(fg=_GREEN if ok else _RED)
        if self._test_btn:
            self._test_btn.config(state="normal")
        if ok and self._status_var:
            self._status_var.set("API ключ проверен успешно")

    # ------------------------------------------------------------------
    # Deepgram key test
    # ------------------------------------------------------------------

    def _test_deepgram_key(self) -> None:
        key = self._deepgram_key_entry.get().strip() if self._deepgram_key_entry else ""
        if not key:
            if self._deepgram_status_var:
                self._deepgram_status_var.set("Введите ключ")
            if self._deepgram_status_lbl:
                self._deepgram_status_lbl.config(fg=_YELLOW)
            return

        if self._deepgram_status_var:
            self._deepgram_status_var.set("Проверяю...")
        if self._deepgram_status_lbl:
            self._deepgram_status_lbl.config(fg=_TEXT_DIM)
        if self._deepgram_test_btn:
            self._deepgram_test_btn.config(state="disabled")

        def _check():
            ok, msg = _check_deepgram_key(key)
            if self._root:
                self._root.after(0, lambda: self._show_deepgram_result(ok, msg))

        threading.Thread(target=_check, daemon=True).start()

    def _show_deepgram_result(self, ok: bool, msg: str) -> None:
        prefix = "\u2713 " if ok else "\u2717 "
        if self._deepgram_status_var:
            self._deepgram_status_var.set(prefix + msg)
        if self._deepgram_status_lbl:
            self._deepgram_status_lbl.config(fg=_GREEN if ok else _RED)
        if self._deepgram_test_btn:
            self._deepgram_test_btn.config(state="normal")
        if ok and self._status_var:
            self._status_var.set("Deepgram API ключ проверен успешно")

    # ------------------------------------------------------------------
    # Local model check
    # ------------------------------------------------------------------

    def _check_local_model(self) -> None:
        if self._local_status_var:
            self._local_status_var.set("Проверяю...")
        if self._local_download_btn:
            self._local_download_btn.config(state="disabled")

        model = self._local_model_var.get() if self._local_model_var else "base"

        def _check():
            ok, msg = _check_local_faster_whisper(model)
            if self._root:
                self._root.after(0, lambda: self._show_local_result(ok, msg))

        threading.Thread(target=_check, daemon=True).start()

    def _show_local_result(self, ok: bool, msg: str) -> None:
        prefix = "\u2713 " if ok else "\u2717 "
        if self._local_status_var:
            self._local_status_var.set(prefix + msg)
        if self._local_download_btn:
            self._local_download_btn.config(state="normal")
        if ok and self._status_var:
            self._status_var.set("Локальная модель готова к использованию")

    # ------------------------------------------------------------------
    # Hotkey capture
    # ------------------------------------------------------------------

    def _start_hotkey_capture(self) -> None:
        self._recording_hotkey = True
        self._pressed_keys.clear()
        self._captured_hotkey = ""
        if self._capture_status_var:
            self._capture_status_var.set("Нажмите комбинацию клавиш...")
        if self._capture_btn:
            self._capture_btn.config(state="disabled")
        if self._hotkey_display_var:
            self._hotkey_display_var.set("[ ожидаю нажатия... ]")

        if self._root:
            self._root.bind("<KeyPress>", self._on_key_press)
            self._root.bind("<KeyRelease>", self._on_key_release)
            self._root.focus_set()

    def _on_key_press(self, event) -> None:
        if not self._recording_hotkey:
            return
        key = _normalize_tk_key(event)
        if key:
            self._pressed_keys.add(key)
        if self._hotkey_display_var:
            self._hotkey_display_var.set(
                " + ".join(sorted(self._pressed_keys)) if self._pressed_keys else "..."
            )

    def _on_key_release(self, event) -> None:
        if not self._recording_hotkey:
            return

        if self._pressed_keys:
            hotkey = _build_pynput_hotkey(self._pressed_keys)
            self._captured_hotkey = hotkey
            if self._hotkey_display_var:
                self._hotkey_display_var.set(self._format_hotkey_display(hotkey))
            if self._capture_status_var:
                self._capture_status_var.set("Клавиша записана!")
        else:
            if self._capture_status_var:
                self._capture_status_var.set("")

        self._recording_hotkey = False
        self._pressed_keys.clear()
        if self._capture_btn:
            self._capture_btn.config(state="normal")
        if self._root:
            self._root.unbind("<KeyPress>")
            self._root.unbind("<KeyRelease>")

    def _reset_hotkey(self) -> None:
        default = "<ctrl>+<shift>+space"
        self._captured_hotkey = default
        if self._hotkey_display_var:
            self._hotkey_display_var.set(self._format_hotkey_display(default))
        if self._capture_status_var:
            self._capture_status_var.set("Сброшено на Ctrl + Shift + Space")

    def _format_hotkey_display(self, hotkey: str) -> str:
        if not hotkey:
            return "Не задана"
        parts = hotkey.split("+")
        display = []
        for p in parts:
            p = p.strip()
            if p.startswith("<") and p.endswith(">"):
                p = p[1:-1]
            display.append(p.capitalize())
        return " + ".join(display)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _update_summary(self) -> None:
        provider = (self._config.get("stt_providers") or ["openai"])[0]
        provider_labels = {
            "openai":   "OpenAI Whisper",
            "deepgram": "Deepgram Nova-3",
            "local":    "Local faster-whisper",
        }
        provider_label = provider_labels.get(provider, provider)

        # Key status depends on provider
        if provider == "openai":
            key = self._key_entry.get().strip() if self._key_entry else ""
            if not key:
                key = self._config.get("api_key", "")
            key_status = "\u2713 Задан" if key else "\u2717 Не задан (распознавание не будет работать)"
        elif provider == "deepgram":
            key = self._deepgram_key_entry.get().strip() if self._deepgram_key_entry else ""
            if not key:
                key = self._config.get("deepgram_api_key", "")
            key_status = "\u2713 Задан" if key else "\u2717 Не задан (распознавание не будет работать)"
        else:  # local
            model_val = self._local_model_var.get() if self._local_model_var else self._config.get("local_whisper_model", "base")
            key_status = f"модель {model_val}"

        hotkey = self._captured_hotkey or self._config.get("hotkey", "<ctrl>+<shift>+space")
        mode = self._mode_var.get() if self._mode_var else self._config.get("hotkey_mode", "toggle")
        mode_label = "Переключение" if mode == "toggle" else "Удержание"

        autostart = bool(
            self._autostart_var.get() if self._autostart_var is not None
            else self._config.get("auto_start", False)
        )
        autostart_label = "\u2713 Включён" if autostart else "\u2717 Выключен"

        summary = (
            f"Провайдер:        {provider_label}\n"
            f"API ключ:         {key_status}\n"
            f"Горячая клавиша:  {self._format_hotkey_display(hotkey)}\n"
            f"Режим записи:     {mode_label}\n"
            f"Автозапуск:       {autostart_label}\n"
        )
        if self._summary_var:
            self._summary_var.set(summary)

    # ------------------------------------------------------------------
    # Toggle key visibility
    # ------------------------------------------------------------------

    def _toggle_key_visibility(self) -> None:
        if self._key_entry and self._show_key_var:
            show_char = "" if self._show_key_var.get() else "\u2022"
            self._key_entry.config(show=show_char)

    # ------------------------------------------------------------------
    # Widget factory
    # ------------------------------------------------------------------

    def _make_btn(self, parent, text: str, command, secondary: bool = False) -> object:
        tk = self._tk
        bg = _ACCENT if secondary else _HIGHLIGHT
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg, fg=_TEXT,
            font=_FONT,
            relief="flat",
            bd=0,
            padx=14, pady=6,
            cursor="hand2",
            activebackground=_BORDER,
            activeforeground=_TEXT,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (no self, easily testable)
# ---------------------------------------------------------------------------

def _check_openai_key(key: str) -> tuple:
    """Test an OpenAI API key. Returns (ok: bool, message: str)."""
    try:
        import urllib.request, urllib.error, json as _json
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            count = len(data.get("data", []))
            return True, f"Ключ действителен ({count} моделей)"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            try:
                body = _json.loads(exc.read())
                err_msg = body.get("error", {}).get("message", "")
                if "invalid" in err_msg.lower() or "incorrect" in err_msg.lower():
                    return False, "Неверный ключ — проверьте правильность"
                if err_msg:
                    return False, f"Ключ отклонён: {err_msg[:80]}"
            except Exception:
                pass
            return False, "Неверный ключ (401) — проверьте правильность"
        if exc.code == 403:
            return False, "Ключ верный, но нет прав на /v1/models — попробуйте использовать"
        if exc.code == 429:
            return True, "Лимит запросов, но ключ рабочий (429)"
        return False, f"Ошибка HTTP {exc.code}"
    except urllib.error.URLError:
        return False, "Нет интернета — проверьте подключение"
    except Exception as exc:
        return False, f"Ошибка: {exc}"


def _check_deepgram_key(key: str) -> tuple:
    """Test a Deepgram API key. Returns (ok: bool, message: str)."""
    try:
        import urllib.request, urllib.error, json as _json
        # Use the Deepgram projects endpoint to validate the key
        req = urllib.request.Request(
            "https://api.deepgram.com/v1/projects",
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            projects = data.get("projects", [])
            return True, f"Ключ действителен ({len(projects)} проектов)"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Неверный ключ (401 Unauthorized)"
        if exc.code == 403:
            return False, "Доступ запрещён (403 Forbidden)"
        return False, f"Ошибка HTTP {exc.code}"
    except urllib.error.URLError:
        return False, "Нет интернета — проверьте подключение"
    except Exception as exc:
        return False, f"Ошибка: {exc}"


def _check_local_faster_whisper(model_size: str = "base") -> tuple:
    """Check if faster-whisper is installed and the model is available."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return (
            False,
            "faster-whisper не установлен. Установите: pip install faster-whisper"
        )

    # Check if model is cached
    try:
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Model dirs are like models--Systran--faster-whisper-base
        model_dir_name = f"models--Systran--faster-whisper-{model_size}"
        if (cache_dir / model_dir_name).exists():
            return True, f"Модель '{model_size}' уже скачана и готова"
        else:
            return (
                False,
                f"Модель '{model_size}' не найдена в кэше. "
                "Она будет скачана автоматически при первом использовании."
            )
    except Exception as exc:
        return True, f"faster-whisper установлен. Статус кэша неизвестен: {exc}"


def _normalize_tk_key(event) -> Optional[str]:
    """Convert a tkinter KeyPress event keysym to a display name."""
    keysym = event.keysym.lower()
    modifiers = {
        "control_l": "Ctrl", "control_r": "Ctrl",
        "shift_l": "Shift", "shift_r": "Shift",
        "alt_l": "Alt", "alt_r": "Alt",
        "super_l": "Win", "super_r": "Win",
    }
    if keysym in modifiers:
        return modifiers[keysym]
    if len(keysym) == 1:
        return keysym.upper()
    named = {
        "space": "Space", "return": "Enter", "tab": "Tab",
        "escape": "Esc",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
        "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
        "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
    }
    return named.get(keysym, keysym.capitalize())


def _build_pynput_hotkey(keys: set) -> str:
    """Build a pynput-compatible hotkey string from a set of display names."""
    order = {"Ctrl": 0, "Shift": 1, "Alt": 2, "Win": 3}
    mods = sorted([k for k in keys if k in order], key=lambda m: order[m])
    regular = [k for k in keys if k not in order]

    parts = [f"<{m.lower()}>" for m in mods]
    parts += [k.lower() for k in regular]
    return "+".join(parts) if parts else "<ctrl>+<shift>+space"


def _format_hotkey_display_fn(hotkey: str) -> str:
    """Module-level version of the hotkey display formatter (used in tests)."""
    if not hotkey:
        return "Не задана"
    parts = hotkey.split("+")
    display = []
    for p in parts:
        p = p.strip()
        if p.startswith("<") and p.endswith(">"):
            p = p[1:-1]
        display.append(p.capitalize())
    return " + ".join(display)


# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------

def show_about_dialog(parent=None) -> None:
    """Show a small 'About Whisper Voice' dialog."""
    try:
        import tkinter as tk
        own_root = parent is None
        if own_root:
            root = tk.Tk()
            root.withdraw()
        else:
            root = parent

        top = tk.Toplevel(root)
        top.title("О программе")
        top.configure(bg=_BG)
        top.resizable(False, False)
        top.attributes("-topmost", True)

        w, h = 340, 220
        sw = top.winfo_screenwidth()
        sh = top.winfo_screenheight()
        top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        tk.Label(
            top,
            text="Whisper Voice",
            bg=_BG, fg=_TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(pady=(28, 4))

        tk.Label(
            top,
            text=f"Версия {APP_VERSION}",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT,
        ).pack()

        tk.Label(
            top,
            text="Голосовой ввод через OpenAI Whisper API",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
        ).pack(pady=(4, 0))

        tk.Label(
            top,
            text="github.com/arwoxbx24/whisper-voice",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
        ).pack()

        tk.Button(
            top,
            text="Закрыть",
            command=top.destroy,
            bg=_ACCENT, fg=_TEXT,
            font=_FONT,
            relief="flat", bd=0,
            padx=20, pady=6,
            cursor="hand2",
        ).pack(pady=(20, 0))

        if own_root:
            root.mainloop()

    except Exception as exc:
        log.error("show_about_dialog failed: %s", exc)
