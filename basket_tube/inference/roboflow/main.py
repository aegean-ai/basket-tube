"""inference-roboflow GPU service — RF-DETR detection, keypoints, OCR."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="inference-roboflow")

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))

# Share core utilities with CPU API
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from api.src.core.video_registry import resolve_title as resolve_stem
from api.src.core.artifacts import config_key, artifact_path, atomic_write_json

from basket_tube.inference.roboflow.models import (
    get_model, run_detection, run_keypoints, run_ocr,
    PLAYER_DETECTION_MODEL_ID, COURT_KEYPOINT_MODEL_ID,
    JERSEY_OCR_MODEL_ID, OCR_PROMPT,
)


class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect", response_model=InferenceResponse)
async def detect(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", PLAYER_DETECTION_MODEL_ID)
        confidence = req.params.get("confidence", 0.4)
        iou_threshold = req.params.get("iou_threshold", 0.9)
        max_frames = req.params.get("max_frames")

        cfg_params = {"model_id": model_id, "confidence": confidence, "iou_threshold": iou_threshold}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "detections", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        if not video_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error=f"Video not found: {video_path}")

        model = get_model(model_id)
        frame_generator = sv.get_video_frames_generator(str(video_path))
        video_info = sv.VideoInfo.from_video_path(str(video_path))

        frames_data = []
        total_detections = 0

        for idx, frame in enumerate(frame_generator):
            if max_frames and idx >= max_frames:
                break
            result = run_detection(model, frame, confidence, iou_threshold)
            detections = sv.Detections.from_inference(result)

            frame_entry = {
                "frame_index": idx,
                "xyxy": detections.xyxy.tolist(),
                "class_id": detections.class_id.tolist(),
                "confidence": detections.confidence.tolist(),
            }
            frames_data.append(frame_entry)
            total_detections += len(detections)

        output = {
            "_meta": {
                "stage": "detections",
                "config_key": cfg_key,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "n_frames": len(frames_data),
            "n_detections": total_detections,
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": video_info.fps,
                "total_frames": video_info.total_frames,
            },
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


@app.post("/api/keypoints", response_model=InferenceResponse)
async def keypoints(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", COURT_KEYPOINT_MODEL_ID)
        keypoint_confidence = req.params.get("keypoint_confidence", 0.3)
        anchor_confidence = req.params.get("anchor_confidence", 0.5)
        det_config_key = req.upstream_configs.get("detections", "")

        cfg_params = {
            "model_id": model_id,
            "keypoint_confidence": keypoint_confidence,
            "anchor_confidence": anchor_confidence,
            "det_config_key": det_config_key,
        }
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "court", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        model = get_model(model_id)
        frame_generator = sv.get_video_frames_generator(str(video_path))

        frames_data = []
        n_mapped = 0

        for idx, frame in enumerate(frame_generator):
            result = run_keypoints(model, frame, keypoint_confidence)
            key_points = sv.KeyPoints.from_inference(result)

            if len(key_points) > 0:
                confident = key_points.confidence[0] > anchor_confidence
                xy = key_points.xy[0][confident].tolist() if confident.any() else []
                conf = key_points.confidence[0][confident].tolist() if confident.any() else []
                if len(xy) >= 4:
                    n_mapped += 1
            else:
                xy, conf = [], []

            frames_data.append({
                "frame_index": idx,
                "keypoints_xy": xy,
                "keypoints_confidence": conf,
            })

        output = {
            "_meta": {
                "stage": "court",
                "config_key": cfg_key,
                "upstream": {"detections": det_config_key},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "n_frames_mapped": n_mapped,
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Keypoint detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


@app.post("/api/ocr", response_model=InferenceResponse)
async def ocr(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", JERSEY_OCR_MODEL_ID)
        n_consecutive = req.params.get("n_consecutive", 3)
        ocr_interval = req.params.get("ocr_interval", 5)
        track_config_key = req.upstream_configs.get("tracks", "")

        cfg_params = {
            "model_id": model_id,
            "n_consecutive": n_consecutive,
            "ocr_interval": ocr_interval,
            "track_config_key": track_config_key,
        }
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "jerseys", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        # Load tracks
        tracks_path = artifact_path(DATA_DIR, "tracks", track_config_key, stem)
        if not tracks_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Tracks not found")

        tracks_data = json.loads(tracks_path.read_text())
        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        model = get_model(model_id)

        from sports import ConsecutiveValueTracker
        number_validator = ConsecutiveValueTracker(n_consecutive=n_consecutive)

        frame_generator = sv.get_video_frames_generator(str(video_path))

        for idx, frame in enumerate(frame_generator):
            if idx >= len(tracks_data.get("frames", [])):
                break
            if idx % ocr_interval != 0:
                continue

            frame_track = tracks_data["frames"][idx]
            tracker_ids = frame_track.get("tracker_ids", [])
            xyxy_list = frame_track.get("xyxy", [])

            if not tracker_ids or not xyxy_list:
                continue

            frame_h, frame_w = frame.shape[:2]
            for tid, box in zip(tracker_ids, xyxy_list):
                x1, y1, x2, y2 = [int(c) for c in box]
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(frame_w, x2 + 10)
                y2 = min(frame_h, y2 + 10)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (224, 224))
                number_str = run_ocr(model, crop_resized, OCR_PROMPT)
                number_validator.update({tid: number_str})

        # Build final player map
        players = {}
        validated = number_validator.get_all_validated() if hasattr(number_validator, "get_all_validated") else {}
        for tid, num in validated.items():
            players[str(tid)] = str(num)

        output = {
            "_meta": {
                "stage": "jerseys",
                "config_key": cfg_key,
                "upstream": {"tracks": track_config_key},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "players": players,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("OCR failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
