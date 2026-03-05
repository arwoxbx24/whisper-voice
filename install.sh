#!/usr/bin/env bash
set -e

echo "================================================"
echo "  Whisper Voice - Installation (Linux/macOS)"
echo "================================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python 3.9+
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.9"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"; then
    echo "Python $PYTHON_VERSION found."
else
    echo "ERROR: Python $PYTHON_VERSION is too old. Requires 3.9+."
    exit 1
fi

# Step 1: virtual environment
echo
echo "[1/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping."
else
    python3 -m venv venv
fi

# Step 2: activate
echo "[2/4] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# Step 3: install deps
echo "[3/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

# Step 4: create launcher
echo "[4/4] Creating launcher script..."
printf '%s\n' '#!/usr/bin/env bash' \
    'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' \
    'cd "$SCRIPT_DIR"' \
    'source venv/bin/activate' \
    'python -m src.app "$@"' > run.sh
chmod +x run.sh

# Linux: try to install system audio deps if missing
if [[ "$(uname)" == "Linux" ]]; then
    if ! python3 -c "import sounddevice" &>/dev/null 2>&1; then
        echo
        echo "Hint: if sounddevice fails, install system libs:"
        echo "  sudo apt-get install -y libportaudio2 portaudio19-dev"
    fi
fi

echo
echo "================================================"
echo "  Installation complete!"
echo "================================================"
echo
echo "To run:   ./run.sh"
echo "      or: source venv/bin/activate && python -m src.app"
echo
echo "Before first run, set your OpenAI API key:"
echo "  export OPENAI_API_KEY=sk-..."
echo "  or add it to config.json"
echo
