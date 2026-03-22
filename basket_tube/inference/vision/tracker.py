"""Multi-object tracker using supervision's ByteTrack.

ByteTrack is a lightweight, CPU-friendly tracker that associates detection
boxes across frames without requiring SAM2's GPU-heavy video loading.
SAM2 can be added later for per-object segmentation masks on tracked results.
"""

from __future__ import annotations

import numpy as np
import supervision as sv


class PlayerTracker:
    """Wraps supervision ByteTrack for frame-by-frame player tracking."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new frame detections. Returns detections with tracker_id assigned."""
        return self.byte_tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state."""
        self.byte_tracker.reset()
