"""
System tray icon + floating recording indicator for Whisper Voice.

Requirements:
    pip install pystray pillow
    # tkinter ships with Python (apt install python3-tk on Debian/Ubuntu)

Architecture:
    TrayIcon        – pystray system-tray icon with context menu
    RecordingIndicator – tkinter floating borderless window (shows while recording)
    UIController    – façade that owns both and exposes simple start/stop/quit

The indicator window:
  - 200 x 60 px, borderless, always-on-top, semi-transparent
  - Pulsing red dot animated at ~30 fps
  - Audio level meter (horizontal bar driven by level_callback from AudioRecorder)
  - "Stop" button  → triggers on_stop callback
  - "X"    button  → triggers on_cancel callback
  - Positioned bottom-right of screen (near typical system tray)
  - Auto-hides when not recording
"""

from __future__ import annotations

import threading
import math
import time
import logging
from typing import Callable, Optional
import io

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when UIController is actually instantiated
# ---------------------------------------------------------------------------

def _import_tkinter():
    import tkinter as tk
    import tkinter.font as tkfont
    return tk, tkfont


def _import_pystray():
    import pystray
    return pystray


def _import_pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Icon image generation (48x48 microphone — no external assets needed)
# ---------------------------------------------------------------------------

def _make_microphone_image(size: int = 48, bg: str = "transparent") -> "Image.Image":
    """
    Draw a simple microphone icon using Pillow primitives.
    Returns an RGBA PIL image.
    """
    Image, ImageDraw, _ = _import_pil()

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size
    w = s // 2          # mic body width
    h = int(s * 0.45)   # mic body height
    cx = s // 2         # center x
    top = int(s * 0.06)

    # Mic body (rounded rectangle)
    x0 = cx - w // 2
    x1 = cx + w // 2
    y0 = top
    y1 = top + h
    r = w // 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=(220, 50, 50, 255))

    # Stand arc (lower semicircle)
    arc_margin = int(s * 0.10)
    arc_top = int(s * 0.42)
    arc_bot = int(s * 0.72)
    draw.arc(
        [arc_margin, arc_top, s - arc_margin, arc_bot],
        start=180, end=0,
        fill=(80, 80, 80, 255),
        width=max(2, s // 14),
    )

    # Vertical stem
    stem_x = cx
    stem_y0 = int(s * 0.60)
    stem_y1 = int(s * 0.78)
    stem_w = max(2, s // 14)
    draw.rectangle(
        [stem_x - stem_w // 2, stem_y0, stem_x + stem_w // 2, stem_y1],
        fill=(80, 80, 80, 255),
    )

    # Base bar
    base_w = int(s * 0.38)
    base_h = max(2, s // 18)
    base_y = stem_y1
    draw.rectangle(
        [cx - base_w // 2, base_y, cx + base_w // 2, base_y + base_h],
        fill=(80, 80, 80, 255),
    )

    return img


def _make_recording_image(size: int = 48) -> "Image.Image":
    """Mic icon with a red ring — used when recording is active."""
    Image, ImageDraw, _ = _import_pil()
    img = _make_microphone_image(size)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [1, 1, size - 2, size - 2],
        outline=(220, 50, 50, 200),
        width=max(2, size // 16),
    )
    return img


def save_icon_png(path: str, size: int = 48) -> None:
    """Save the microphone icon as a PNG file (helper for external use)."""
    img = _make_microphone_image(size)
    img.save(path)
    log.info("Icon saved to %s", path)


# ---------------------------------------------------------------------------
# RecordingIndicator — floating tkinter window
# ---------------------------------------------------------------------------

class RecordingIndicator:
    """
    Small always-on-top borderless window that appears during recording.

    Must be created and driven from the tkinter main thread.
    Use show()/hide() and update_level() which are thread-safe via tk.after().
    """

    WIN_W = 200
    WIN_H = 60
    ANIM_FPS = 30
    BG = "#1e1e1e"
    TEXT_COLOR = "#eeeeee"
    LEVEL_BAR_COLOR = "#4caf50"
    DOT_ACTIVE = "#e53935"
    DOT_PULSE_MIN = 0.35   # alpha factor min for pulse
    PADDING = 10

    def __init__(
        self,
        tk_root,
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
    ):
        self._root = tk_root
        self._on_stop = on_stop
        self._on_cancel = on_cancel

        self._visible = False
        self._level: float = 0.0          # 0.0–1.0
        self._pulse_phase: float = 0.0    # 0.0–2π
        self._anim_running = False

        self._win: Optional[object] = None  # tk.Toplevel
        self._canvas = None
        self._dot_id = None
        self._bar_id = None
        self._bar_bg_id = None
        self._stop_btn = None
        self._cancel_btn = None

        self._build()

    # ------------------------------------------------------------------ build

    def _build(self):
        tk, tkfont = _import_tkinter()

        win = tk.Toplevel(self._root)
        win.title("")
        win.overrideredirect(True)          # borderless
        win.attributes("-topmost", True)    # always on top
        win.attributes("-alpha", 0.92)      # slight transparency
        win.configure(bg=self.BG)
        win.resizable(False, False)

        # Position bottom-right (near tray)
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = sw - self.WIN_W - 12
        y = sh - self.WIN_H - 48
        win.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")

        # Canvas for pulsing dot + level bar
        canvas = tk.Canvas(
            win,
            width=self.WIN_W,
            height=self.WIN_H,
            bg=self.BG,
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)

        # --- Pulsing dot ---
        dot_r = 8
        dot_cx = self.PADDING + dot_r
        dot_cy = self.WIN_H // 2 - 8
        dot_id = canvas.create_oval(
            dot_cx - dot_r, dot_cy - dot_r,
            dot_cx + dot_r, dot_cy + dot_r,
            fill=self.DOT_ACTIVE, outline="",
        )

        # --- "REC" label ---
        canvas.create_text(
            dot_cx + dot_r + 6, dot_cy,
            text="REC", anchor="w",
            fill=self.DOT_ACTIVE,
            font=("Helvetica", 8, "bold"),
        )

        # --- Level bar background ---
        bar_x0 = self.PADDING
        bar_y0 = dot_cy + dot_r + 6
        bar_x1 = self.WIN_W - self.PADDING - 60  # leave room for buttons
        bar_y1 = bar_y0 + 6
        bar_bg_id = canvas.create_rectangle(
            bar_x0, bar_y0, bar_x1, bar_y1,
            fill="#444444", outline="",
        )
        bar_id = canvas.create_rectangle(
            bar_x0, bar_y0, bar_x0, bar_y1,   # zero width initially
            fill=self.LEVEL_BAR_COLOR, outline="",
        )
        self._bar_x0 = bar_x0
        self._bar_x1_max = bar_x1
        self._bar_y0 = bar_y0
        self._bar_y1 = bar_y1

        # --- Buttons (placed as canvas windows) ---
        btn_y = self.WIN_H // 2 - 1
        btn_x_stop = self.WIN_W - 56
        btn_x_x = self.WIN_W - 18

        stop_btn = tk.Button(
            win, text="Stop",
            command=self._handle_stop,
            bg="#2979ff", fg="white",
            relief="flat", bd=0,
            padx=4, pady=1,
            font=("Helvetica", 8, "bold"),
            cursor="hand2",
        )
        canvas.create_window(btn_x_stop, btn_y, window=stop_btn, anchor="center")

        cancel_btn = tk.Button(
            win, text="X",
            command=self._handle_cancel,
            bg="#555555", fg="white",
            relief="flat", bd=0,
            padx=4, pady=1,
            font=("Helvetica", 8, "bold"),
            cursor="hand2",
        )
        canvas.create_window(btn_x_x, btn_y, window=cancel_btn, anchor="center")

        # Allow dragging the window
        canvas.bind("<ButtonPress-1>", self._drag_start)
        canvas.bind("<B1-Motion>", self._drag_motion)
        self._drag_x = 0
        self._drag_y = 0

        # --- Countdown label (shown when max_recording_seconds > 0) ---
        self._countdown_id = canvas.create_text(
            self.WIN_W // 2, self.WIN_H - 8,
            text="",
            fill="#aaaaaa",
            font=("Helvetica", 7),
            anchor="center",
        )

        self._win = win
        self._canvas = canvas
        self._dot_id = dot_id
        self._bar_id = bar_id
        self._bar_bg_id = bar_bg_id
        self._stop_btn = stop_btn
        self._cancel_btn = cancel_btn
        self._dot_cx = dot_cx
        self._dot_cy = dot_cy
        self._dot_r = dot_r

        win.withdraw()   # hidden by default

    # ----------------------------------------------------------------- drag

    def _drag_start(self, event):
        self._drag_x = event.x_root - self._win.winfo_x()
        self._drag_y = event.y_root - self._win.winfo_y()

    def _drag_motion(self, event):
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self._win.geometry(f"+{x}+{y}")

    # -------------------------------------------------------------- callbacks

    def _handle_stop(self):
        self.hide()
        try:
            self._on_stop()
        except Exception:
            log.exception("Error in on_stop callback")

    def _handle_cancel(self):
        self.hide()
        try:
            self._on_cancel()
        except Exception:
            log.exception("Error in on_cancel callback")

    # ------------------------------------------------------------- public API

    def show(self):
        """Show the indicator window. Thread-safe."""
        self._win.after(0, self._do_show)

    def hide(self):
        """Hide the indicator window. Thread-safe."""
        self._win.after(0, self._do_hide)

    def update_level(self, level: float):
        """Update audio level bar. Thread-safe. level ∈ [0.0, 1.0]."""
        self._level = max(0.0, min(1.0, level))

    def update_countdown(self, remaining: Optional[int]):
        """Update countdown text. Thread-safe. remaining=None clears the label."""
        self._win.after(0, self._do_update_countdown, remaining)

    def _do_update_countdown(self, remaining: Optional[int]):
        """Must run in tk thread."""
        if remaining is None or remaining <= 0:
            self._canvas.itemconfig(self._countdown_id, text="")
            return
        minutes = remaining // 60
        seconds = remaining % 60
        text = f"авто-стоп {minutes:02d}:{seconds:02d}"
        # Turn red when less than 30 seconds left
        color = "#e53935" if remaining <= 30 else "#aaaaaa"
        self._canvas.itemconfig(self._countdown_id, text=text, fill=color)

    # ----------------------------------------------------------- internal ops

    def _do_show(self):
        self._visible = True
        self._win.deiconify()
        self._win.lift()
        if not self._anim_running:
            self._anim_running = True
            self._animate()

    def _do_hide(self):
        self._visible = False
        self._anim_running = False
        self._win.withdraw()

    def _animate(self):
        """Animation loop — called via tk.after every ~33 ms."""
        if not self._anim_running:
            return

        dt = 1.0 / self.ANIM_FPS
        self._pulse_phase = (self._pulse_phase + dt * 2.0) % (2 * math.pi)

        # Pulse: brightness oscillates between DOT_PULSE_MIN and 1.0
        alpha = self.DOT_PULSE_MIN + (1.0 - self.DOT_PULSE_MIN) * (
            0.5 + 0.5 * math.sin(self._pulse_phase)
        )
        r = int(0xe5 * alpha)
        g = int(0x39 * alpha)
        b = int(0x35 * alpha)
        dot_color = f"#{r:02x}{g:02x}{b:02x}"
        self._canvas.itemconfig(self._dot_id, fill=dot_color)

        # Level bar width
        bar_w = int((self._bar_x1_max - self._bar_x0) * self._level)
        bar_x1 = self._bar_x0 + bar_w
        # Color transitions green→yellow→red with level
        if self._level < 0.6:
            bar_color = self.LEVEL_BAR_COLOR   # green
        elif self._level < 0.85:
            bar_color = "#ffc107"              # yellow
        else:
            bar_color = "#e53935"              # red (clipping)
        self._canvas.coords(
            self._bar_id,
            self._bar_x0, self._bar_y0,
            bar_x1, self._bar_y1,
        )
        self._canvas.itemconfig(self._bar_id, fill=bar_color)

        self._win.after(int(1000 / self.ANIM_FPS), self._animate)


# ---------------------------------------------------------------------------
# TrayIcon — pystray wrapper
# ---------------------------------------------------------------------------

class TrayIcon:
    """
    Manages the system-tray icon using pystray.

    Runs pystray in its own daemon thread (pystray.Icon.run() blocks).
    Callbacks are invoked from that thread — ensure they're thread-safe.
    """

    def __init__(
        self,
        on_settings: Callable[[], None],
        on_about: Callable[[], None],
        on_quit: Callable[[], None],
        on_toggle_recording: Optional[Callable[[], None]] = None,
    ):
        self._on_settings = on_settings
        self._on_about = on_about
        self._on_quit = on_quit
        self._on_toggle = on_toggle_recording
        self._icon: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._recording = False

    def start(self) -> None:
        """Spawn pystray icon in a background thread."""
        pystray = _import_pystray()
        Image, ImageDraw, _ = _import_pil()

        img_normal = _make_microphone_image(48)

        def _settings(_icon, _item): self._on_settings()
        def _about(_icon, _item): self._on_about()
        def _quit(_icon, _item):
            _icon.stop()
            self._on_quit()

        menu_items = [
            pystray.MenuItem("Whisper Voice", lambda *a: None, enabled=False),
            pystray.Menu.SEPARATOR,
        ]
        if self._on_toggle:
            def _toggle(_icon, _item): self._on_toggle()
            menu_items.append(
                pystray.MenuItem("Начать / Остановить запись", _toggle, default=True)
            )
            menu_items.append(pystray.Menu.SEPARATOR)

        menu_items += [
            pystray.MenuItem("Настройки", _settings),
            pystray.MenuItem("О программе", _about),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", _quit),
        ]

        self._icon = pystray.Icon(
            name="whisper-voice",
            icon=img_normal,
            title="Whisper Voice",
            menu=pystray.Menu(*menu_items),
        )

        self._thread = threading.Thread(
            target=self._icon.run,
            name="pystray-thread",
            daemon=True,
        )
        self._thread.start()
        log.info("TrayIcon started")

    def set_recording(self, recording: bool) -> None:
        """Update tray icon to reflect recording state."""
        if self._icon is None:
            return
        self._recording = recording
        img = _make_recording_image(48) if recording else _make_microphone_image(48)
        self._icon.icon = img
        self._icon.title = "Whisper Voice — Recording" if recording else "Whisper Voice"

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UIController — public façade
# ---------------------------------------------------------------------------

class UIController:
    """
    Top-level UI controller.  Owns the tk root, the tray icon, and the
    recording indicator.  Call run() from the main thread — it blocks until
    the user quits.

    Callbacks provided by the application layer:
        on_stop_recording()   – user pressed "Stop" in indicator
        on_cancel_recording() – user pressed "X" in indicator
        on_settings()         – user chose Settings from tray menu
        on_about()            – user chose About from tray menu
        on_quit()             – user chose Quit (also called on window close)
        on_toggle_recording() – optional, double-click / tray default action

    Thread safety:
        All tk operations are dispatched via tk.after().
        Callbacks may be called from pystray's thread or the tk thread.
    """

    def __init__(
        self,
        on_stop_recording: Callable[[], None],
        on_cancel_recording: Callable[[], None],
        on_settings: Optional[Callable[[], None]] = None,
        on_about: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        on_toggle_recording: Optional[Callable[[], None]] = None,
    ):
        self._cb_stop = on_stop_recording
        self._cb_cancel = on_cancel_recording
        self._cb_settings = on_settings or (lambda: None)
        self._cb_about = on_about or (lambda: None)
        self._cb_quit = on_quit or (lambda: None)
        self._cb_toggle = on_toggle_recording

        self._recording = False
        self._tray: Optional[TrayIcon] = None
        self._indicator: Optional[RecordingIndicator] = None
        self._root = None   # set in run()

    # ------------------------------------------------------------ lifecycle

    def run(self) -> None:
        """
        Initialise tkinter, build the indicator, start the tray icon,
        then enter the tk event loop.  Blocks until quit() is called.
        """
        tk, _ = _import_tkinter()

        root = tk.Tk()
        root.withdraw()             # main window invisible — we only use Toplevel
        root.title("Whisper Voice")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._root = root

        # Build indicator
        self._indicator = RecordingIndicator(
            tk_root=root,
            on_stop=self._handle_stop,
            on_cancel=self._handle_cancel,
        )

        # Build tray icon
        self._tray = TrayIcon(
            on_settings=self._cb_settings,
            on_about=self._handle_about,
            on_quit=self._handle_quit,
            on_toggle_recording=self._cb_toggle,
        )
        try:
            self._tray.start()
        except Exception as exc:
            log.warning("Could not start tray icon: %s (continuing without tray)", exc)

        log.info("UIController running")
        root.mainloop()

    def quit(self) -> None:
        """Quit the UI event loop and tray icon. Thread-safe."""
        if self._root is not None:
            self._root.after(0, self._do_quit)

    def show_recording(self) -> None:
        """Show the floating indicator to signal recording has started. Thread-safe."""
        self._recording = True
        if self._tray:
            self._tray.set_recording(True)
        if self._indicator:
            self._indicator.show()

    def hide_recording(self) -> None:
        """Hide the floating indicator. Thread-safe."""
        self._recording = False
        if self._tray:
            self._tray.set_recording(False)
        if self._indicator:
            self._indicator.hide()

    def update_audio_level(self, level: float) -> None:
        """
        Feed audio level to the indicator bar.
        Designed to be used as the level_callback for AudioRecorder.
        Thread-safe.
        """
        if self._indicator:
            self._indicator.update_level(level)

    def update_countdown(self, remaining: Optional[int]) -> None:
        """Update the recording countdown in the indicator. Thread-safe."""
        if self._indicator:
            self._indicator.update_countdown(remaining)

    def show_notification(self, message: str) -> None:
        """Show a brief notification in the recording indicator title area. Thread-safe."""
        if self._root:
            self._root.after(0, lambda: self._do_show_notification(message))

    def _do_show_notification(self, message: str) -> None:
        """Show notification (tk thread). Uses a simple Toplevel toast."""
        try:
            tk, _ = _import_tkinter()
            toast = tk.Toplevel(self._root)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            toast.configure(bg="#323232")
            sw = toast.winfo_screenwidth()
            sh = toast.winfo_screenheight()
            w, h = 320, 36
            x = sw // 2 - w // 2
            y = sh - 120
            toast.geometry(f"{w}x{h}+{x}+{y}")
            lbl = tk.Label(
                toast, text=message,
                bg="#323232", fg="#ffffff",
                font=("Helvetica", 10),
                padx=12, pady=6,
            )
            lbl.pack(fill="both", expand=True)
            # Auto-close after 3 seconds
            toast.after(3000, toast.destroy)
        except Exception as exc:
            log.debug("Notification display error: %s", exc)

    def is_recording(self) -> bool:
        return self._recording

    # ----------------------------------------------------------- handlers

    def _handle_stop(self):
        self.hide_recording()
        try:
            self._cb_stop()
        except Exception:
            log.exception("Error in stop callback")

    def _handle_cancel(self):
        self.hide_recording()
        try:
            self._cb_cancel()
        except Exception:
            log.exception("Error in cancel callback")

    def _handle_about(self):
        try:
            self._cb_about()
        except Exception:
            log.exception("Error in about callback")

    def _handle_quit(self):
        self.quit()
        try:
            self._cb_quit()
        except Exception:
            log.exception("Error in quit callback")

    def _do_quit(self):
        if self._tray:
            try:
                self._tray.stop()
            except Exception:
                pass
        if self._root:
            self._root.quit()
            self._root.destroy()
            self._root = None


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    import threading as _threading

    logging.basicConfig(level=logging.DEBUG)

    def _fake_recording_loop(ui: UIController):
        """Simulate 5 seconds of recording with a sine-wave audio level."""
        _time.sleep(1.5)  # wait for UI to settle
        print("[test] starting fake recording")
        ui.show_recording()
        t0 = _time.time()
        while _time.time() - t0 < 5:
            elapsed = _time.time() - t0
            level = 0.5 + 0.45 * math.sin(elapsed * 3)
            ui.update_audio_level(level)
            _time.sleep(0.05)
        print("[test] fake recording done — hiding indicator")
        ui.hide_recording()
        _time.sleep(2)
        print("[test] quitting")
        ui.quit()

    ctrl = UIController(
        on_stop_recording=lambda: print("[cb] STOP"),
        on_cancel_recording=lambda: print("[cb] CANCEL"),
        on_settings=lambda: print("[cb] SETTINGS"),
        on_about=lambda: print("[cb] ABOUT"),
        on_quit=lambda: print("[cb] QUIT"),
    )

    t = _threading.Thread(target=_fake_recording_loop, args=(ctrl,), daemon=True)
    t.start()

    ctrl.run()   # blocks until quit
    print("[test] bye")
