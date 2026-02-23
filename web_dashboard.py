#!/usr/bin/env python3
"""
core/input_sources.py — Input source abstraction.

Provides a unified frame iterator for:
  - USB/V4L2 camera (GStreamer v4l2src)
  - CSI camera (GStreamer nvarguscamerasrc)
  - RTSP stream (GStreamer rtspsrc + hw decode)
  - Video file (GStreamer filesrc + hw decode)

All sources yield numpy BGR frames (HxWx3 uint8).

Falls back to OpenCV VideoCapture for simple camera/file if GStreamer pipeline fails.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger("input_sources")


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class FrameSource(ABC):
    """Abstract base: iterate to get BGR numpy frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        ...

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# GStreamer helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_gst_pipeline_str(input_cfg: dict) -> str:
    """Build a GStreamer pipeline string that ends with appsink."""
    itype = input_cfg.get("type", "camera")
    uri = input_cfg.get("uri", 0)
    w = input_cfg.get("width", 1280)
    h = input_cfg.get("height", 720)
    use_csi = input_cfg.get("use_csi", False)
    sensor_id = input_cfg.get("sensor_id", 0)

    caps = f"video/x-raw,format=BGR,width={w},height={h}"
    appsink = "appsink name=sink emit-signals=false max-buffers=2 drop=true"

    if itype == "camera":
        if use_csi:
            # CSI camera via nvarguscamerasrc (Jetson-specific)
            src = (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM),width={w},height={h},framerate=30/1 ! "
                f"nvvidconv ! "
                f"video/x-raw,format=BGRx ! "
                f"videoconvert ! "
                f"{caps} ! "
                f"{appsink}"
            )
        else:
            # USB/V4L2 camera
            dev = f"/dev/video{uri}" if isinstance(uri, int) else uri
            src = (
                f"v4l2src device={dev} ! "
                f"video/x-raw,width={w},height={h},framerate=30/1 ! "
                f"videoconvert ! "
                f"{caps} ! "
                f"{appsink}"
            )
    elif itype == "rtsp":
        # RTSP with hardware decode
        src = (
            f"rtspsrc location={uri} latency=100 ! "
            f"rtph264depay ! h264parse ! "
            f"nvv4l2decoder ! "
            f"nvvidconv ! "
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"{caps} ! "
            f"{appsink}"
        )
    elif itype == "video":
        ext = str(uri).lower()
        if ext.endswith(".mp4") or ext.endswith(".mkv") or ext.endswith(".mov"):
            src = (
                f"filesrc location={uri} ! "
                f"decodebin ! "
                f"nvvidconv ! "
                f"video/x-raw,format=BGRx ! "
                f"videoconvert ! "
                f"{caps} ! "
                f"{appsink}"
            )
        else:
            raise ValueError(f"Unsupported video format: {uri}")
    else:
        raise ValueError(f"Unknown input type: {itype}")

    return src


# ──────────────────────────────────────────────────────────────────────────────
# GStreamer source
# ──────────────────────────────────────────────────────────────────────────────

class GStreamerSource(FrameSource):
    """
    Opens a GStreamer pipeline and yields BGR frames via OpenCV's VideoCapture
    with GStreamer backend.
    """

    def __init__(self, input_cfg: dict):
        self.input_cfg = input_cfg
        self.pipeline_str = _build_gst_pipeline_str(input_cfg)
        log.debug(f"GStreamer pipeline: {self.pipeline_str}")
        self._cap = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open GStreamer pipeline.\n"
                f"Pipeline: {self.pipeline_str}\n"
                f"Check camera device, RTSP URL, or file path."
            )
        return cap

    def __iter__(self) -> Iterator[np.ndarray]:
        self._cap = self._open()
        log.info(f"GStreamer source opened: {self.input_cfg['type']} / {self.input_cfg['uri']}")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    log.warning("GStreamer: read returned False — end of stream or error.")
                    break
                yield frame
        finally:
            self._cap.release()
            self._cap = None

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ──────────────────────────────────────────────────────────────────────────────
# OpenCV fallback source
# ──────────────────────────────────────────────────────────────────────────────

class OpenCVSource(FrameSource):
    """
    Fallback source using OpenCV VideoCapture (no GStreamer).
    Works for USB cameras and video files.
    """

    def __init__(self, input_cfg: dict):
        self.input_cfg = input_cfg
        self._cap = None

    def _get_index_or_uri(self):
        itype = self.input_cfg.get("type", "camera")
        uri = self.input_cfg.get("uri", 0)
        if itype == "camera":
            return int(uri)
        elif itype in ("rtsp", "video"):
            return str(uri)
        raise ValueError(f"OpenCV fallback does not support type={itype}")

    def __iter__(self) -> Iterator[np.ndarray]:
        src = self._get_index_or_uri()
        self._cap = cv2.VideoCapture(src)

        w = self.input_cfg.get("width", 1280)
        h = self.input_cfg.get("height", 720)
        if isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open: {src}")

        log.info(f"OpenCV fallback source opened: {src}")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    log.info("OpenCV: end of stream.")
                    break

                # Resize to target resolution if needed
                fh, fw = frame.shape[:2]
                if fw != w or fh != h:
                    frame = cv2.resize(frame, (w, h))

                yield frame
        finally:
            self._cap.release()
            self._cap = None

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_source(input_cfg: dict) -> FrameSource:
    """
    Build and return the appropriate FrameSource.
    Tries GStreamer first; falls back to OpenCV on error.
    """
    itype = input_cfg.get("type", "camera")

    # Always try GStreamer first (preferred on Jetson)
    try:
        src = GStreamerSource(input_cfg)
        # Probe by opening the cap briefly
        cap = cv2.VideoCapture(src.pipeline_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            cap.release()
            log.info("Using GStreamer source.")
            return src
        else:
            cap.release()
            raise RuntimeError("GStreamer pipeline probe failed.")
    except Exception as gst_exc:
        log.warning(f"GStreamer source failed: {gst_exc}")

        # Fallback: OpenCV (camera / video file only)
        if itype in ("camera", "video", "rtsp"):
            log.info("Falling back to OpenCV VideoCapture.")
            return OpenCVSource(input_cfg)
        else:
            raise RuntimeError(
                f"Cannot open input (type={itype}, uri={input_cfg.get('uri')}).\n"
                f"GStreamer error: {gst_exc}"
            )
