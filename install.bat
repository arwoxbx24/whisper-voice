@echo off
setlocal enabledelayedexpansion

echo ================================================
echo   Whisper Voice - Installation
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [3/4] Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Creating launcher...
REM run.bat is already present in the repo

echo.
echo ================================================
echo   Installation complete!
echo ================================================
echo.
echo To run:   double-click run.bat
echo      or   python main.py
echo.
echo Before first run, set your OpenAI API key in:
echo   %%USERPROFILE%%\.whisper-voice\config.json
echo or set environment variable: OPENAI_API_KEY=sk-...
echo.
pause
