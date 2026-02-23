#!/usr/bin/env python3
"""
core/perf.py — FPS and latency measurement with rolling window.
"""

import time
from collections import deque


class PerfMonitor:
    """
    Measures:
      - FPS: rolling average over `window` frames
      - Per-frame latency (ms): time between tick_start and tick_end
    """

    def __init__(self, window: int = 30):
        self.window = window
        self._frame_times: deque = deque(maxlen=window)
        self._start_time: float | None = None
        self.fps: float = 0.0
        self.last_latency_ms: float = 0.0

    def tick_start(self):
        """Call at the start of frame processing."""
        self._start_time = time.perf_counter()

    def tick_end(self):
        """Call at the end of frame processing."""
        now = time.perf_counter()
        if self._start_time is not None:
            latency = (now - self._start_time) * 1000.0  # ms
            self.last_latency_ms = round(latency, 2)
            self._frame_times.append(now)
            self._update_fps()

    def _update_fps(self):
        if len(self._frame_times) < 2:
            self.fps = 0.0
            return
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed > 0:
            self.fps = round((len(self._frame_times) - 1) / elapsed, 1)

    def summary(self) -> dict:
        return {
            "fps": self.fps,
            "latency_ms": self.last_latency_ms,
        }
