#!/usr/bin/env bash
# scripts/install_deps.sh — Install all dependencies for Jetson Vision Suite
# Run as: sudo ./scripts/install_deps.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================="
echo " Jetson Vision Suite — Dependency Installer"
echo "======================================================="
echo ""

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/7] Installing system packages ..."

apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-rtsp \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    v4l-utils \
    wget \
    git \
    build-essential \
    cmake

echo "[1/7] System packages installed."

# ── 2. DeepStream Python bindings ─────────────────────────────────────────────
echo "[2/7] Checking DeepStream Python bindings ..."

DS_PYDS_PATH="/opt/nvidia/deepstream/deepstream/lib/pyds.so"
if [ -f "$DS_PYDS_PATH" ]; then
    # Add to Python path
    PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    SITE_PKGS="/usr/lib/python3/dist-packages"
    if [ ! -f "$SITE_PKGS/pyds.so" ]; then
        echo "  Linking pyds.so to Python site-packages ..."
        ln -sf "$DS_PYDS_PATH" "$SITE_PKGS/pyds.so" || true
    fi
    echo "[2/7] DeepStream pyds found."
else
    echo "[2/7] WARNING: DeepStream not found at $DS_PYDS_PATH"
    echo "  Install DeepStream SDK from: https://developer.nvidia.com/deepstream-sdk"
    echo "  Segmentation mode will fall back to direct TensorRT inference."
fi

# Also check for DeepStream GStreamer plugins
if ! gst-inspect-1.0 nvinfer >/dev/null 2>&1; then
    echo "  WARNING: nvinfer GStreamer plugin not found."
    echo "  Check DeepStream installation and PATH."
fi

# ── 3. Python virtualenv ───────────────────────────────────────────────────────
echo "[3/7] Creating Python virtualenv ..."

VENV_DIR="$PROJECT_DIR/.venv"
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── 4. Python requirements ────────────────────────────────────────────────────
echo "[4/7] Installing Python requirements ..."
pip install -r "$PROJECT_DIR/requirements.txt" -q

# ── 5. NanoOWL ────────────────────────────────────────────────────────────────
echo "[5/7] Installing NanoOWL ..."

NANOOWL_DIR="/opt/nanoowl"
if [ ! -d "$NANOOWL_DIR" ]; then
    echo "  Cloning NanoOWL from GitHub ..."
    git clone https://github.com/NVIDIA-AI-IOT/nanoowl.git "$NANOOWL_DIR"
else
    echo "  NanoOWL already cloned at $NANOOWL_DIR"
fi

cd "$NANOOWL_DIR"
pip install -e . -q
echo "[5/7] NanoOWL installed."

cd "$PROJECT_DIR"

# ── 6. PyCUDA (required for TRT fallback) ─────────────────────────────────────
echo "[6/7] Installing pycuda ..."
pip install pycuda -q || echo "  WARNING: pycuda install failed. TRT direct inference may not work."

# ── 7. Create output directories ──────────────────────────────────────────────
echo "[7/7] Creating output directories ..."
mkdir -p "$PROJECT_DIR/outputs/logs" "$PROJECT_DIR/outputs/videos"

# ── 8. Model directories ───────────────────────────────────────────────────────
echo "  Creating model placeholder directories ..."
mkdir -p /data/models/nanoowl /data/models/segment
echo "  ⚠  Place your TRT engine files in:"
echo "     /data/models/nanoowl/owl_image_encoder_patch32.engine  (NanoOWL)"
echo "     /data/models/segment/peoplsegnet.engine                 (Segmentation)"

echo ""
echo "======================================================="
echo " Installation complete!"
echo ""
echo " Next steps:"
echo "  1. Build NanoOWL engine:"
echo "     source .venv/bin/activate"
echo "     python3 -m nanoowl.build_image_encoder_engine /data/models/nanoowl/owl_image_encoder_patch32.engine"
echo ""
echo "  2. Build segmentation engine (see README.md)"
echo ""
echo "  3. Run: ./run.sh"
echo "======================================================="
