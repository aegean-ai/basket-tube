"""SAM2-based multi-object tracker. Extracted from notebook cell 42."""

from __future__ import annotations

import numpy as np
import torch
import supervision as sv


class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")
        if detections.tracker_id is None:
            detections.tracker_id = list(range(1, len(detections) + 1))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for tid, box in zip(detections.tracker_id, detections.xyxy):
                _, _, _ = self.predictor.add_new_prompt(
                    frame_idx=0, obj_id=int(tid), bbox=box.tolist(),
                )
        self._prompted = True

    def track(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame() before track()")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            obj_ids, mask_logits = self.predictor.track(frame)
        if len(obj_ids) == 0:
            return sv.Detections.empty()
        masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            tracker_id=np.array(obj_ids, dtype=int),
        )
