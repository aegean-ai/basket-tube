"""BasketTube GPU inference service — all vision models in one FastAPI app.

Endpoints:
  /api/detect          — RF-DETR player detection
  /api/keypoints       — Court keypoint detection
  /api/ocr             — Jersey number OCR
  /api/track           — SAM2 segmentation + tracking
  /api/classify-teams  — SigLIP team classification
  /health              — Health check
"""

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

app = FastAPI(title="basket-tube-inference")

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))
SAM2_REPO = os.environ.get("SAM2_REPO", "/opt/segment-anything-2-real-time")

# Share core utilities with CPU API
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from api.src.video_registry import resolve_title as resolve_stem
from api.src.artifacts import config_key, artifact_path, atomic_write_json

from basket_tube.inference.roboflow.models import (
    get_model, run_detection, run_keypoints, run_ocr,
    PLAYER_DETECTION_MODEL_ID, COURT_KEYPOINT_MODEL_ID,
    JERSEY_OCR_MODEL_ID, OCR_PROMPT,
)


# ── Schemas ───────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None


PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]


# ── Health ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Detection (RF-DETR) ──────────────────────────────────────────────

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

            frames_data.append({
                "frame_index": idx,
                "xyxy": detections.xyxy.tolist(),
                "class_id": detections.class_id.tolist(),
                "confidence": detections.confidence.tolist(),
            })
            total_detections += len(detections)

        output = {
            "_meta": {"stage": "detections", "config_key": cfg_key, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames": len(frames_data),
            "n_detections": total_detections,
            "video_info": {"width": video_info.width, "height": video_info.height, "fps": video_info.fps, "total_frames": video_info.total_frames},
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── Keypoints (court detection) ───────────────────────────────────────

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

        cfg_params = {"model_id": model_id, "keypoint_confidence": keypoint_confidence, "anchor_confidence": anchor_confidence, "det_config_key": det_config_key}
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

            frames_data.append({"frame_index": idx, "keypoints_xy": xy, "keypoints_confidence": conf})

        output = {
            "_meta": {"stage": "court", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames_mapped": n_mapped,
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Keypoint detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── OCR (jersey numbers) ─────────────────────────────────────────────

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

        cfg_params = {"model_id": model_id, "n_consecutive": n_consecutive, "ocr_interval": ocr_interval, "track_config_key": track_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "jerseys", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

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
                x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
                x2, y2 = min(frame_w, x2 + 10), min(frame_h, y2 + 10)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (224, 224))
                number_str = run_ocr(model, crop_resized, OCR_PROMPT)
                number_validator.update({tid: number_str})

        players = {}
        validated = number_validator.get_all_validated() if hasattr(number_validator, "get_all_validated") else {}
        for tid, num in validated.items():
            players[str(tid)] = str(num)

        output = {
            "_meta": {"stage": "jerseys", "config_key": cfg_key, "upstream": {"tracks": track_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "players": players,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("OCR failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── Tracking (SAM2) ──────────────────────────────────────────────────

@app.post("/api/track", response_model=InferenceResponse)
async def track(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        det_config_key = req.upstream_configs.get("detections", "")
        cfg_params = {"sam2_checkpoint": "sam2.1_hiera_large.pt", "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "tracks", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        if not det_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Detections not found")
        det_data = json.loads(det_path.read_text())

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"

        sys.path.insert(0, SAM2_REPO)
        from sam2.build_sam import build_sam2_camera_predictor
        checkpoint = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        old_cwd = os.getcwd()
        os.chdir(SAM2_REPO)
        predictor = build_sam2_camera_predictor(sam2_config, checkpoint)
        os.chdir(old_cwd)

        from basket_tube.inference.vision.tracker import SAM2Tracker
        tracker = SAM2Tracker(predictor)

        frame_generator = sv.get_video_frames_generator(str(video_path))
        max_frames = req.params.get("max_frames")

        frames_data = []
        all_tracker_ids = set()

        for idx, frame in enumerate(frame_generator):
            if max_frames and idx >= max_frames:
                break

            if idx == 0:
                first_frame = det_data["frames"][0]
                xyxy = np.array(first_frame["xyxy"])
                class_ids = np.array(first_frame["class_id"])
                player_mask = np.isin(class_ids, PLAYER_CLASS_IDS)

                if player_mask.any():
                    player_xyxy = xyxy[player_mask]
                    initial = sv.Detections(xyxy=player_xyxy, class_id=class_ids[player_mask])
                    initial.tracker_id = np.arange(1, len(initial) + 1)
                    tracker.prompt_first_frame(frame, initial)

                    frames_data.append({
                        "frame_index": idx,
                        "tracker_ids": initial.tracker_id.tolist(),
                        "xyxy": initial.xyxy.tolist(),
                        "mask_rle": [],
                    })
                    all_tracker_ids.update(initial.tracker_id.tolist())
                    continue

            tracked = tracker.track(frame)
            frames_data.append({
                "frame_index": idx,
                "tracker_ids": tracked.tracker_id.tolist() if tracked.tracker_id is not None else [],
                "xyxy": tracked.xyxy.tolist(),
                "mask_rle": [],
            })
            if tracked.tracker_id is not None:
                all_tracker_ids.update(tracked.tracker_id.tolist())

        output = {
            "_meta": {"stage": "tracks", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames": len(frames_data),
            "n_tracks": len(all_tracker_ids),
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Tracking failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── Team Classification (SigLIP) ─────────────────────────────────────

@app.post("/api/classify-teams", response_model=InferenceResponse)
async def classify_teams(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        stride = req.params.get("stride", 30)
        crop_scale = req.params.get("crop_scale", 0.4)
        det_config_key = req.upstream_configs.get("detections", "")

        cfg_params = {"stride": stride, "crop_scale": crop_scale, "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "teams", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        if not det_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Detections not found")
        det_data = json.loads(det_path.read_text())

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"

        from basket_tube.inference.vision.classifier import extract_player_crops
        from sports import TeamClassifier

        crops = []
        crop_metadata = []
        frame_generator = sv.get_video_frames_generator(str(video_path), stride=stride)

        for frame_idx_sampled, frame in enumerate(frame_generator):
            actual_idx = frame_idx_sampled * stride
            if actual_idx >= len(det_data["frames"]):
                break

            frame_det = det_data["frames"][actual_idx]
            xyxy = np.array(frame_det["xyxy"])
            class_ids = np.array(frame_det["class_id"])
            player_mask = np.isin(class_ids, PLAYER_CLASS_IDS)

            if not player_mask.any():
                continue

            player_xyxy = xyxy[player_mask]
            player_indices = np.where(player_mask)[0]

            scaled = sv.scale_boxes(xyxy=player_xyxy, factor=crop_scale)
            for box, det_idx in zip(scaled, player_indices):
                crop = sv.crop_image(frame, box)
                if crop.size > 0:
                    crops.append(crop)
                    crop_metadata.append({"frame_index": actual_idx, "detection_index": int(det_idx)})

        if len(crops) < 2:
            output = {
                "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}},
                "palette": {"0": {"name": "Team A", "color": "#006BB6"}, "1": {"name": "Team B", "color": "#007A33"}},
                "assignments": [],
            }
            atomic_write_json(out, output)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        classifier = TeamClassifier(device="cuda")
        classifier.fit(crops)
        teams = classifier.predict(crops)

        assignments = [
            {"frame_index": meta["frame_index"], "detection_index": meta["detection_index"], "team_id": int(team_id)}
            for meta, team_id in zip(crop_metadata, teams)
        ]

        output = {
            "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "palette": {"0": {"name": "Team A", "color": "#006BB6"}, "1": {"name": "Team B", "color": "#007A33"}},
            "assignments": assignments,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Team classification failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
