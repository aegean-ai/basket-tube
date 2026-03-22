"""SAM2-based multi-object tracker using Meta's official SAM-2 package."""

from __future__ import annotations

import numpy as np
import torch
import supervision as sv


def build_tracker(model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml", checkpoint: str | None = None):
    """Build a SAM2VideoPredictor from Meta's official package.

    The model config and checkpoint are resolved automatically by sam2.
    """
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    return predictor


class SAM2Tracker:
    """Wraps SAM2VideoPredictor for frame-by-frame tracking with supervision Detections."""

    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._inference_state = None
        self._prompted = False

    def init_video(self, video_path: str) -> None:
        """Initialize tracking state from a video file.

        Offloads video frames and state to CPU RAM to avoid GPU OOM
        on long videos. Frames are moved to GPU only during processing.
        """
        self._inference_state = self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )

    def prompt_frame(self, frame_idx: int, detections: sv.Detections) -> None:
        """Add bounding box prompts for objects in a specific frame."""
        if self._inference_state is None:
            raise RuntimeError("Call init_video() before prompt_frame()")
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")
        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1)

        for tid, box in zip(detections.tracker_id, detections.xyxy):
            self.predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=int(tid),
                box=box.tolist(),
            )
        self._prompted = True

    def propagate(self) -> list[dict]:
        """Propagate tracking through all video frames.

        Returns a list of per-frame results:
        [{"frame_index": int, "tracker_ids": list, "xyxy": list, "mask_rle": list}, ...]
        """
        if not self._prompted:
            raise RuntimeError("Call prompt_frame() before propagate()")

        frames_data = []
        all_tracker_ids = set()

        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            self._inference_state
        ):
            if len(obj_ids) == 0:
                frames_data.append({
                    "frame_index": frame_idx,
                    "tracker_ids": [],
                    "xyxy": [],
                    "mask_rle": [],
                })
                continue

            masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
            xyxy = sv.mask_to_xyxy(masks)

            tracker_ids = [int(oid) for oid in obj_ids]
            all_tracker_ids.update(tracker_ids)

            frames_data.append({
                "frame_index": frame_idx,
                "tracker_ids": tracker_ids,
                "xyxy": xyxy.tolist(),
                "mask_rle": [],
            })

        return frames_data, len(all_tracker_ids)
