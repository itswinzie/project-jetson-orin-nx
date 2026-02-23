#!/usr/bin/env bash
# scripts/benchmark.sh — Benchmark both modes and report FPS / latency
# Usage: ./scripts/benchmark.sh [test_video.mp4]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_DIR/.venv/bin/python3"
DURATION=60  # seconds per test

# Test video: use argument or find a sample
TEST_VIDEO="${1:-}"
if [ -z "$TEST_VIDEO" ]; then
    # Try to find a test video
    TEST_VIDEO=$(find /tmp /home -name "*.mp4" 2>/dev/null | head -1)
    if [ -z "$TEST_VIDEO" ]; then
        echo "No test video found. Creating a synthetic test video ..."
        TEST_VIDEO="/tmp/benchmark_test.mp4"
        ffmpeg -f lavfi -i testsrc=duration=60:size=1280x720:rate=30 \
               -c:v libx264 -preset ultrafast "$TEST_VIDEO" -y -q 2>/dev/null
    fi
fi

echo "======================================================="
echo " Jetson Vision Suite — Benchmark"
echo " Test video: $TEST_VIDEO"
echo " Duration:   ${DURATION}s per mode"
echo "======================================================="
echo ""

run_bench() {
    local mode="$1"
    local label="$2"

    echo "--- Benchmarking: $label ---"
    local log_file="/tmp/bench_${mode}.jsonl"
    rm -f "$log_file"

    timeout "$DURATION" "$PYTHON" apps/main.py \
        --mode "$mode" \
        --input "video:$TEST_VIDEO" \
        --no-web \
        2>/dev/null || true

    # Parse FPS from log files
    local log_dir="$PROJECT_DIR/outputs/logs"
    local latest_log=$(ls -t "$log_dir"/${mode}_*.jsonl 2>/dev/null | head -1)

    if [ -n "$latest_log" ]; then
        echo "  Log: $latest_log"
        python3 -c "
import json, sys
fps_vals = []
lat_vals = []
with open('$latest_log') as f:
    for line in f:
        try:
            d = json.loads(line)
            if 'fps' in d: fps_vals.append(d['fps'])
            if 'latency_ms' in d: lat_vals.append(d['latency_ms'])
        except: pass

if fps_vals:
    print(f'  FPS:     avg={sum(fps_vals)/len(fps_vals):.1f}  min={min(fps_vals):.1f}  max={max(fps_vals):.1f}  (n={len(fps_vals)})')
if lat_vals:
    print(f'  Latency: avg={sum(lat_vals)/len(lat_vals):.1f}ms  min={min(lat_vals):.1f}ms  max={max(lat_vals):.1f}ms')
" 2>/dev/null || echo "  (Could not parse log)"
    else
        echo "  (No log file generated)"
    fi
    echo ""
}

cd "$PROJECT_DIR"

run_bench "detect_open_vocab" "Mode 1: NanoOWL Detection"
run_bench "segment" "Mode 2: DeepStream Segmentation"

echo "======================================================="
echo " Benchmark complete."
echo " Full logs: $PROJECT_DIR/outputs/logs/"
echo "======================================================="
