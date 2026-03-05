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
        "sounddevice",
        "soundfile",
        "numpy",
        "openai",
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._win32",
        "pynput.mouse",
        "pynput.mouse._win32",
        "pyperclip",
        "pystray",
        "pystray._win32",
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFont",
        "pkg_resources.py2_compat",
        "httpx",
        "httpcore",
        "anyio",
        "certifi",
        "charset_normalizer",
        "deepgram",
        "tkinter",
        "tkinter.font",
        "win32api",
        "win32con",
        "win32gui",
    ]


def _data_files():
    files = [f"{ROOT / 'src'}{SEP}src"]
    assets_dir = ROOT / "assets"
    if assets_dir.exists():
        files.append(f"{assets_dir}{SEP}assets")
    return files


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
