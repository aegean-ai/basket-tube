"""Team classification wrapper using sports.TeamClassifier."""

import numpy as np
import supervision as sv


def extract_player_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    player_class_ids: list[int],
    crop_scale: float = 0.4,
) -> list[np.ndarray]:
    """Extract scaled center crops of players from a frame."""
    player_mask = np.isin(detections.class_id, player_class_ids)
    player_dets = detections[player_mask]
    boxes = sv.scale_boxes(xyxy=player_dets.xyxy, factor=crop_scale)
    return [sv.crop_image(frame, box) for box in boxes]
