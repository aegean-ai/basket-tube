"""inference-vision GPU service — SAM2 tracking, TeamClassifier."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import supervision as sv
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="inference-vision")

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))
SAM2_REPO = os.environ.get("SAM2_REPO", "/opt/segment-anything-2-real-time")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.src.core.video_registry import resolve_title as resolve_stem
from api.src.core.artifacts import config_key, artifact_path, atomic_write_json


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


@app.get("/health")
async def health():
    return {"status": "ok"}


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

        # Load SAM2
        sys.path.insert(0, SAM2_REPO)
        from sam2.build_sam import build_sam2_camera_predictor
        checkpoint = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        old_cwd = os.getcwd()
        os.chdir(SAM2_REPO)
        predictor = build_sam2_camera_predictor(sam2_config, checkpoint)
        os.chdir(old_cwd)

        from inference_vision.tracker import SAM2Tracker
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
                    initial = sv.Detections(
                        xyxy=player_xyxy,
                        class_id=class_ids[player_mask],
                    )
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
            "_meta": {
                "stage": "tracks",
                "config_key": cfg_key,
                "upstream": {"detections": det_config_key},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "n_frames": len(frames_data),
            "n_tracks": len(all_tracker_ids),
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Tracking failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


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

        from inference_vision.classifier import extract_player_crops
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

        assignments = []
        for meta, team_id in zip(crop_metadata, teams):
            assignments.append({
                "frame_index": meta["frame_index"],
                "detection_index": meta["detection_index"],
                "team_id": int(team_id),
            })

        output = {
            "_meta": {
                "stage": "teams",
                "config_key": cfg_key,
                "upstream": {"detections": det_config_key},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "palette": {
                "0": {"name": "Team A", "color": "#006BB6"},
                "1": {"name": "Team B", "color": "#007A33"},
            },
            "assignments": assignments,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Team classification failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
