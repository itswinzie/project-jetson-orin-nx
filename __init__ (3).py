#!/usr/bin/env python3
"""
core/tracker.py — Lightweight IoU-based object tracker.

Implements a simple SORT-inspired tracker:
  - Associates detections to tracks via IoU matching (Hungarian algorithm).
  - Maintains track IDs across frames.
  - No Kalman filter by default (add filterpy if needed for smoother tracking).
"""

import logging
from typing import List

import numpy as np

log = logging.getLogger("tracker")


def _iou(boxA: list, boxB: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    aA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    aB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aA + aB - inter

    return inter / union if union > 0 else 0.0


def _hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """
    Greedy matching (approximate Hungarian) for small matrices.
    Returns list of (det_idx, track_idx) pairs.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matches = []
    unmatched_dets = list(range(cost_matrix.shape[0]))
    unmatched_tracks = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            matches.append((r, c))
            unmatched_dets.remove(r)
            unmatched_tracks.remove(c)

    return matches, unmatched_dets, unmatched_tracks


class Track:
    _id_counter = 0

    def __init__(self, det: dict):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = det["bbox"]
        self.label = det.get("label", "?")
        self.confidence = det.get("confidence", 0.0)
        self.hits = 1
        self.age = 0

    def update(self, det: dict):
        self.bbox = det["bbox"]
        self.label = det.get("label", self.label)
        self.confidence = det.get("confidence", self.confidence)
        self.hits += 1
        self.age = 0

    def mark_missed(self):
        self.age += 1

    def to_det_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "track_id": self.id,
        }


class IoUTracker:
    """
    Simple IoU tracker compatible with NanoOWL detections.

    Args:
        iou_threshold: minimum IoU to associate detection to track
        max_age: max frames a track can be unmatched before deletion
        min_hits: minimum hits before a track is confirmed (returned in output)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update tracker with new detections.
        Returns list of dicts with track_id assigned.
        """
        if not detections:
            # Age all tracks
            for t in self.tracks:
                t.mark_missed()
            self._prune()
            return []

        if not self.tracks:
            # Initialize tracks from first detections
            for det in detections:
                self.tracks.append(Track(det))
            self._prune()
            return [t.to_det_dict() for t in self.tracks if t.hits >= self.min_hits]

        # Build IoU cost matrix: (n_dets, n_tracks)
        n_dets = len(detections)
        n_tracks = len(self.tracks)
        iou_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                # Only match same label (open-vocab: labels may differ)
                if det.get("label") == track.label:
                    iou_matrix[i, j] = _iou(det["bbox"], track.bbox)

        matches, unmatched_dets, unmatched_tracks = _hungarian_match(
            iou_matrix, self.iou_threshold
        )

        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # Age unmatched tracks
        for trk_idx in unmatched_tracks:
            self.tracks[trk_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(Track(detections[det_idx]))

        self._prune()

        # Return confirmed tracks
        return [t.to_det_dict() for t in self.tracks if t.hits >= self.min_hits]

    def _prune(self):
        """Remove dead tracks."""
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

    def reset(self):
        self.tracks = []
        Track._id_counter = 0
