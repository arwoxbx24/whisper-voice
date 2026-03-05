"""
Setup Wizard for Whisper Voice.

Opens automatically on first launch (no config.json) or when user
requests it via tray menu → "Настройки".

Steps:
  1. Добро пожаловать  — описание приложения
  2. API ключ          — ввод OpenAI API key + тест
  3. Горячая клавиша   — выбор комбинации
  4. Готово!           — кнопка запуска

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

    NUM_STEPS = 4

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
        win_w, win_h = 520, 460
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
        """Step 2 — API Key."""
        tk = self._tk
        frame = tk.Frame(parent, bg=_BG)

        tk.Label(
            frame,
            text="Настройка API ключа",
            bg=_BG, fg=_TEXT,
            font=_FONT_BIG,
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            frame,
            text="Получить ключ: platform.openai.com/api-keys",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # Key entry row
        entry_frame = tk.Frame(frame, bg=_BG2, highlightthickness=1, highlightbackground=_BORDER)
        entry_frame.pack(fill="x", ipady=2)

        tk.Label(
            entry_frame,
            text="sk-",
            bg=_BG2, fg=_TEXT_DIM,
            font=_FONT,
        ).pack(side="left", padx=(8, 0))

        self._key_entry = tk.Entry(
            entry_frame,
            bg=_BG2, fg=_TEXT,
            font=_FONT,
            relief="flat",
            highlightthickness=0,
            show="\u2022",
            insertbackground=_TEXT,
        )
        self._key_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(2, 8))

        # Pre-fill if key exists
        existing_key = self._config.get("api_key", "")
        if existing_key:
            self._key_entry.insert(0, existing_key)

        # Show/hide toggle
        self._show_key_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            entry_frame,
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

        # Test button + status
        btn_row = tk.Frame(frame, bg=_BG)
        btn_row.pack(fill="x", pady=(10, 0))

        self._test_btn = self._make_btn(btn_row, "Проверить ключ", self._test_api_key)
        self._test_btn.pack(side="left")

        self._api_status_var = tk.StringVar(value="")
        self._api_status_lbl = tk.Label(
            btn_row,
            textvariable=self._api_status_var,
            bg=_BG,
            fg=_GREEN,
            font=_FONT,
        )
        self._api_status_lbl.pack(side="left", padx=(12, 0))

        # Cost hint
        tk.Label(
            frame,
            text="\nСтоимость: ~$0.006 за минуту (~0.5 руб).\nКлюч хранится только локально.",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            frame,
            text="\nМожно пропустить и ввести ключ позже через Настройки.",
            bg=_BG, fg=_TEXT_DIM,
            font=_FONT_SMALL,
            justify="left",
        ).pack(anchor="w")

        return frame

    def _build_step2(self, parent) -> object:
        """Step 3 — Hotkey."""
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

    def _build_step3(self, parent) -> object:
        """Step 4 — Done."""
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
            "Шаг 1 из 4 — Добро пожаловать",
            "Шаг 2 из 4 — API ключ",
            "Шаг 3 из 4 — Горячая клавиша",
            "Шаг 4 из 4 — Готово",
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

        if self._status_var:
            self._status_var.set("")

    def _go_next(self) -> None:
        if self._current_step == 1:
            # Save API key
            key = self._key_entry.get().strip() if self._key_entry else ""
            if key:
                self._config["api_key"] = key
        elif self._current_step == 2:
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
        # Final API key
        if self._key_entry:
            key = self._key_entry.get().strip()
            if key:
                self._config["api_key"] = key

        # Final hotkey
        if self._captured_hotkey:
            self._config["hotkey"] = self._captured_hotkey
        if self._mode_var:
            self._config["hotkey_mode"] = self._mode_var.get()

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
        key = ""
        if self._key_entry:
            key = self._key_entry.get().strip()
        if not key:
            key = self._config.get("api_key", "")

        has_key = bool(key)
        hotkey = self._captured_hotkey or self._config.get("hotkey", "<ctrl>+<shift>+space")
        mode = self._mode_var.get() if self._mode_var else self._config.get("hotkey_mode", "toggle")
        mode_label = "Переключение" if mode == "toggle" else "Удержание"
        key_status = "\u2713 Задан" if has_key else "\u2717 Не задан (распознавание не будет работать)"

        summary = (
            f"API ключ:         {key_status}\n"
            f"Горячая клавиша:  {self._format_hotkey_display(hotkey)}\n"
            f"Режим записи:     {mode_label}\n"
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
            return False, "Неверный ключ (401 Unauthorized)"
        if exc.code == 429:
            return True, "Лимит запросов, но ключ рабочий (429)"
        return False, f"Ошибка HTTP {exc.code}"
    except urllib.error.URLError:
        return False, "Нет интернета — проверьте подключение"
    except Exception as exc:
        return False, f"Ошибка: {exc}"


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
