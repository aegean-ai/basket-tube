"""Roboflow model loading with local/remote mode switching."""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "local")

PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
COURT_KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"
JERSEY_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."


@lru_cache
def get_model(model_id: str):
    """Load a Roboflow model (cached)."""
    logger.info("Loading model %s (mode=%s)", model_id, INFERENCE_MODE)
    from inference import get_model as rf_get_model
    return rf_get_model(model_id=model_id)


def run_detection(model, frame, confidence: float = 0.4, iou_threshold: float = 0.9):
    """Run detection on a single frame."""
    return model.infer(frame, confidence=confidence, iou_threshold=iou_threshold)[0]


def run_keypoints(model, frame, confidence: float = 0.3):
    """Run keypoint detection on a single frame."""
    return model.infer(frame, confidence=confidence)[0]


def run_ocr(model, crop, prompt: str = OCR_PROMPT):
    """Run OCR on a single crop."""
    return model.predict(crop, prompt)[0]
