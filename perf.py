#!/usr/bin/env bash
# run.sh — Quick launcher for Jetson Vision Suite
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python3"

# ── 1. Create virtualenv if missing ────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[run.sh] Creating virtualenv at $VENV_DIR ..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# ── 2. Install / upgrade Python requirements ──────────────────────────────────
echo "[run.sh] Installing Python requirements ..."
"$PYTHON" -m pip install --upgrade pip -q
"$PYTHON" -m pip install -r requirements.txt -q

# ── 3. Ensure output directories exist ────────────────────────────────────────
mkdir -p outputs/logs outputs/videos

# ── 4. Parse CLI args (pass-through to main.py) ───────────────────────────────
MODE="${MODE:-detect_open_vocab}"
INPUT="${INPUT:-camera:0}"
EXTRA_ARGS="$@"

echo "[run.sh] Starting Jetson Vision Suite"
echo "  Mode  : $MODE"
echo "  Input : $INPUT"
echo "  Web UI: http://$(hostname -I | awk '{print $1}'):8080"
echo ""

# ── 5. Launch pipeline + web dashboard ────────────────────────────────────────
# Web dashboard runs in background; main.py runs in foreground.
# Both share state via the events module (in-process threads when using same Python process).
# For production, use --no-web and run web_dashboard.py separately.

exec "$PYTHON" apps/main.py \
    --mode "$MODE" \
    --input "$INPUT" \
    --web \
    $EXTRA_ARGS
