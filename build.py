"""
build.py — PyInstaller script to create a standalone Windows .exe for Whisper Voice.

Usage:
    python build.py              # single-file exe (default, same as --onefile)
    python build.py --onefile    # single-file exe
    python build.py --portable   # onedir folder with exe + all DLLs

Output:
    --onefile:  dist/WhisperVoice.exe
    --portable: dist/WhisperVoice/ (folder)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Windows uses ';' as path separator in --add-data, Linux/macOS use ':'
SEP = ";" if sys.platform == "win32" else ":"


def _locate_icon():
    """Find an icon file in common locations, return path string or None."""
    icon_candidates = [
        ROOT / "assets" / "icon.ico",
        ROOT / "src" / "icon.ico",
        ROOT / "assets" / "icon.png",
        ROOT / "src" / "icon.png",
        ROOT / "src" / "icon_128.png",
    ]
    for candidate in icon_candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _hidden_imports():
    return [
        # Audio
        "sounddevice",
        "soundfile",
        "numpy",
        "wave",
        "cffi",
        "_cffi_backend",
        # OpenAI
        "openai",
        "openai.types",
        "openai.types.audio",
        "openai._utils",
        "openai._models",
        # HTTP stack
        "httpx",
        "httpcore",
        "httpcore._backends.sync",
        "httpcore._backends.anyio",
        "httpcore._backends.trio",
        "anyio",
        "anyio._backends._asyncio",
        "anyio._backends._trio",
        "certifi",
        "charset_normalizer",
        "idna",
        "h11",
        # Hotkey / input
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._win32",
        "pynput.mouse",
        "pynput.mouse._win32",
        # Clipboard
        "pyperclip",
        "win32clipboard",
        "win32con",
        "win32api",
        "win32gui",
        "pywintypes",
        # Tray icon
        "pystray",
        "pystray._win32",
        # Pillow
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFont",
        "PIL._imaging",
        # Tkinter (bundled with Python, but PyInstaller needs hints)
        "tkinter",
        "tkinter.font",
        "tkinter.messagebox",
        "tkinter.ttk",
        "_tkinter",
        # Deepgram SDK (optional provider)
        "deepgram",
        "deepgram.audio",
        "deepgram.clients",
        "deepgram.clients.listen",
        "websockets",
        "websockets.client",
        "websockets.connection",
        "aiohttp",
        # Standard library modules PyInstaller sometimes misses
        "sqlite3",
        "_sqlite3",
        "socket",
        "ssl",
        "_ssl",
        "email",
        "email.mime",
        "email.mime.text",
        "email.mime.multipart",
        "html",
        "html.parser",
        "urllib",
        "urllib.parse",
        "urllib.request",
        "http",
        "http.client",
        "json",
        "pathlib",
        "ctypes",
        "ctypes.util",
        "_ctypes",
        "threading",
        "queue",
        "concurrent",
        "concurrent.futures",
        "asyncio",
        # pkg_resources / importlib
        "pkg_resources",
        "pkg_resources.py2_compat",
        "importlib.metadata",
        # anyio helper
        "sniffio",
    ]


def _data_files():
    files = [f"{ROOT / 'src'}{SEP}src"]
    assets_dir = ROOT / "assets"
    if assets_dir.exists():
        files.append(f"{assets_dir}{SEP}assets")
    return files


def _collect_all_packages():
    """Packages that use dynamic imports and need --collect-all."""
    return [
        "openai",
        "pystray",
        "pynput",
        "PIL",
        "sounddevice",
        "soundfile",
        "deepgram",
        "httpx",
        "httpcore",
        "anyio",
        "pyperclip",
    ]


def _run_pyinstaller(mode_flag: str, icon_path: str | None):
    """Build with PyInstaller. mode_flag is '--onefile' or '--onedir'."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        mode_flag,
        "--windowed",
        "--name", "WhisperVoice",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT),
        "--clean",
        "--noconfirm",
    ]

    if icon_path:
        cmd += ["--icon", icon_path]

    for hidden in _hidden_imports():
        cmd += ["--hidden-import", hidden]

    for pkg in _collect_all_packages():
        cmd += ["--collect-all", pkg]

    for data in _data_files():
        cmd += ["--add-data", data]

    cmd.append(str(ROOT / "main.py"))

    print(f"Running PyInstaller ({mode_flag}):")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\nBuild FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)


def build_onefile():
    """Build a single-file exe: dist/WhisperVoice.exe"""
    icon_path = _locate_icon()
    _run_pyinstaller("--onefile", icon_path)
    exe = ROOT / "dist" / "WhisperVoice.exe"
    print(f"\nBuild successful: {exe}")


def build_portable():
    """Build onedir portable folder: dist/WhisperVoice/"""
    icon_path = _locate_icon()
    _run_pyinstaller("--onedir", icon_path)

    # Copy README-INSTALL.txt into the portable folder
    readme_src = ROOT / "README-INSTALL.txt"
    portable_dir = ROOT / "dist" / "WhisperVoice"
    if readme_src.exists() and portable_dir.exists():
        shutil.copy2(readme_src, portable_dir / "README-INSTALL.txt")
        print(f"Copied README-INSTALL.txt -> {portable_dir / 'README-INSTALL.txt'}")

    print(f"\nPortable build successful: {portable_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WhisperVoice for Windows")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--onefile",
        action="store_true",
        default=False,
        help="Build single-file exe (default)",
    )
    group.add_argument(
        "--portable",
        action="store_true",
        default=False,
        help="Build portable onedir folder with exe + all DLLs",
    )
    args = parser.parse_args()

    if args.portable:
        build_portable()
    else:
        # --onefile is default (both explicit and no args)
        build_onefile()
