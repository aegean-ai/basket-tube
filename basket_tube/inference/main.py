"""BasketTube GPU inference service — all vision models in one FastAPI app.

Endpoints:
  /api/detect          — RF-DETR player detection
  /api/keypoints       — Court keypoint detection
  /api/ocr             — Jersey number OCR
  /api/track           — SAM2 segmentation + tracking
  /api/classify-teams  — SigLIP team classification
  /health              — Health check

Instrumented with Pydantic Logfire for deep observability.
"""

import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Logfire setup ─────────────────────────────────────────────────────
try:
    import logfire
    _token = os.environ.get("LOGFIRE_TOKEN", "")
    if _token:
        logfire.configure(token=_token, service_name="basket-tube-inference")
        _logfire = True
    else:
        _logfire = False
except ImportError:
    logfire = None  # type: ignore
    _logfire = False

app = FastAPI(title="basket-tube-inference")

if _logfire:
    logfire.instrument_fastapi(app)

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))

# Share core utilities with CPU API
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from api.src.video_registry import resolve_title as resolve_stem
from api.src.artifacts import config_key, artifact_path, atomic_write_json

from basket_tube.inference.progress import write_progress

from basket_tube.inference.roboflow.models import (
    get_model, run_detection, run_keypoints, run_ocr,
    PLAYER_DETECTION_MODEL_ID, COURT_KEYPOINT_MODEL_ID,
    JERSEY_OCR_MODEL_ID, OCR_PROMPT,
)

# Class name mapping for readable logs
CLASS_NAMES = {
    0: "ball", 1: "ball-in-basket", 2: "number", 3: "player",
    4: "player-possession", 5: "jump-shot", 6: "layup-dunk",
    7: "shot-block", 8: "referee", 9: "rim",
}

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

# ── Helpers ───────────────────────────────────────────────────────────

LOG_INTERVAL = 100  # log progress every N frames


def _log_info(msg: str, **kwargs):
    if _logfire:
        logfire.info(msg, **kwargs)
    logger.info(msg.format(**kwargs) if kwargs else msg)


def _class_distribution(class_ids: list[int]) -> dict[str, int]:
    """Human-readable class distribution."""
    counts = Counter(class_ids)
    return {CLASS_NAMES.get(k, f"class_{k}"): v for k, v in counts.items()}


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
            _log_info("detect.cache_hit video_id={video_id} config_key={config_key}", video_id=req.video_id, config_key=cfg_key)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        if not video_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error=f"Video not found: {video_path}")

        span = logfire.span("detect.process", video_id=req.video_id, model_id=model_id, confidence=confidence) if _logfire else None
        if span:
            span.__enter__()

        try:
            _log_info("detect.model_loading model_id={model_id}", model_id=model_id)
            t0 = time.monotonic()
            model = get_model(model_id)
            _log_info("detect.model_loaded elapsed_s={elapsed:.1f}", elapsed=time.monotonic() - t0)

            frame_generator = sv.get_video_frames_generator(str(video_path))
            video_info = sv.VideoInfo.from_video_path(str(video_path))
            total_frames = video_info.total_frames or 0
            _log_info("detect.video_info width={w} height={h} fps={fps} total_frames={total}",
                       w=video_info.width, h=video_info.height, fps=video_info.fps, total=total_frames)

            frames_data = []
            total_detections = 0
            t_start = time.monotonic()

            for idx, frame in enumerate(frame_generator):
                if max_frames and idx >= max_frames:
                    break
                result = run_detection(model, frame, confidence, iou_threshold)
                detections = sv.Detections.from_inference(result)

                n_det = len(detections)
                frames_data.append({
                    "frame_index": idx,
                    "xyxy": detections.xyxy.tolist(),
                    "class_id": detections.class_id.tolist(),
                    "confidence": detections.confidence.tolist(),
                })
                total_detections += n_det

                if idx % 10 == 0:
                    write_progress(out.parent, frame=idx, total_frames=total_frames)

                if idx > 0 and idx % LOG_INTERVAL == 0:
                    fps = idx / (time.monotonic() - t_start)
                    pct = (idx / total_frames * 100) if total_frames else 0
                    dist = _class_distribution(detections.class_id.tolist())
                    _log_info("detect.progress frame={frame}/{total} ({pct:.0f}%) detections={n_det} fps={fps:.1f} classes={classes}",
                              frame=idx, total=total_frames, pct=pct, n_det=n_det, fps=fps, classes=dist)

            elapsed = time.monotonic() - t_start
            avg_fps = len(frames_data) / elapsed if elapsed > 0 else 0
            _log_info("detect.complete frames={n_frames} detections={n_det} elapsed_s={elapsed:.1f} avg_fps={fps:.1f}",
                       n_frames=len(frames_data), n_det=total_detections, elapsed=elapsed, fps=avg_fps)

            output = {
                "_meta": {"stage": "detections", "config_key": cfg_key, "created_at": datetime.now(timezone.utc).isoformat()},
                "n_frames": len(frames_data),
                "n_detections": total_detections,
                "video_info": {"width": video_info.width, "height": video_info.height, "fps": video_info.fps, "total_frames": video_info.total_frames},
                "frames": frames_data,
            }

            atomic_write_json(out, output)
            (out.parent / "_progress.json").unlink(missing_ok=True)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("Detection failed")
        if _logfire:
            logfire.error("detect.failed error={error}", error=str(e))
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

        span = logfire.span("keypoints.process", video_id=req.video_id, model_id=model_id) if _logfire else None
        if span:
            span.__enter__()

        try:
            model = get_model(model_id)
            frame_generator = sv.get_video_frames_generator(str(video_path))
            kp_video_info = sv.VideoInfo.from_video_path(str(video_path))
            kp_total_frames = kp_video_info.total_frames or 0

            frames_data = []
            n_mapped = 0
            t_start = time.monotonic()

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

                if idx % 10 == 0:
                    write_progress(out.parent, frame=idx, total_frames=kp_total_frames)

                if idx > 0 and idx % LOG_INTERVAL == 0:
                    _log_info("keypoints.progress frame={frame} mapped={mapped} keypoints_this_frame={n_kp}",
                              frame=idx, mapped=n_mapped, n_kp=len(xy))

            elapsed = time.monotonic() - t_start
            _log_info("keypoints.complete frames={n_frames} mapped={mapped} elapsed_s={elapsed:.1f}",
                       n_frames=len(frames_data), mapped=n_mapped, elapsed=elapsed)

            output = {
                "_meta": {"stage": "court", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
                "n_frames_mapped": n_mapped,
                "frames": frames_data,
            }

            atomic_write_json(out, output)
            (out.parent / "_progress.json").unlink(missing_ok=True)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("Keypoint detection failed")
        if _logfire:
            logfire.error("keypoints.failed error={error}", error=str(e))
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

        span = logfire.span("ocr.process", video_id=req.video_id, model_id=model_id, ocr_interval=ocr_interval) if _logfire else None
        if span:
            span.__enter__()

        try:
            tracks_data = json.loads(tracks_path.read_text())
            det_config_key = req.upstream_configs.get("detections", "")
            det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem) if det_config_key else None

            video_path = DATA_DIR / "videos" / f"{stem}.mp4"
            ocr_model = get_model(model_id)
            detection_model = get_model(PLAYER_DETECTION_MODEL_ID)

            from sports import ConsecutiveValueTracker
            number_validator = ConsecutiveValueTracker(n_consecutive=n_consecutive)

            NUMBER_CLASS_ID = 2  # "number" class from RF-DETR

            frame_generator = sv.get_video_frames_generator(str(video_path))
            ocr_total_frames = len(tracks_data.get("frames", []))
            n_ocr_frames = 0
            n_ocr_reads = 0
            t_start = time.monotonic()

            for idx, frame in enumerate(frame_generator):
                if idx >= ocr_total_frames:
                    break
                if idx % ocr_interval != 0:
                    continue

                if idx % 10 == 0:
                    write_progress(out.parent, frame=idx, total_frames=ocr_total_frames)

                frame_track = tracks_data["frames"][idx]
                tracker_ids = frame_track.get("tracker_ids", [])
                player_xyxy = frame_track.get("xyxy", [])

                if not tracker_ids or not player_xyxy:
                    continue

                n_ocr_frames += 1
                frame_h, frame_w = frame.shape[:2]

                # Step 1: Run RF-DETR to find number regions (class_id=2)
                det_result = run_detection(detection_model, frame, 0.4, 0.9)
                all_dets = sv.Detections.from_inference(det_result)
                number_dets = all_dets[all_dets.class_id == NUMBER_CLASS_ID]

                if len(number_dets) == 0:
                    continue

                # Step 2: OCR only the detected number crops
                number_crops = [
                    sv.crop_image(frame, xyxy)
                    for xyxy in sv.clip_boxes(
                        sv.pad_boxes(xyxy=number_dets.xyxy, px=10, py=10),
                        (frame_w, frame_h),
                    )
                ]
                numbers = [
                    run_ocr(ocr_model, cv2.resize(crop, (224, 224)), OCR_PROMPT)
                    for crop in number_crops if crop.size > 0
                ]
                n_ocr_reads += len(numbers)

                # Step 3: Match numbers to players via IoS (mask overlap)
                player_boxes = np.array(player_xyxy)
                player_masks = sv.xyxy_to_mask(boxes=player_boxes, resolution_wh=(frame_w, frame_h))
                number_masks = sv.xyxy_to_mask(boxes=number_dets.xyxy, resolution_wh=(frame_w, frame_h))

                iou = sv.mask_iou_batch(
                    masks_true=player_masks,
                    masks_detection=number_masks,
                    overlap_metric=sv.OverlapMetric.IOS,
                )

                # Find pairs above IoS threshold 0.9
                rows, cols = np.where(iou > 0.9)
                if len(rows) > 0:
                    matched_tids = [tracker_ids[int(r)] for r in rows]
                    matched_numbers = [numbers[int(c)] for c in cols if int(c) < len(numbers)]
                    if matched_tids and matched_numbers:
                        number_validator.update(matched_tids[:len(matched_numbers)], matched_numbers)

                if n_ocr_frames % 50 == 0:
                    validated_so_far = number_validator.validated_dict()
                    _log_info("ocr.progress frame={frame} ocr_frames={n_ocr} reads={reads} number_dets={n_num} validated={n_val}",
                              frame=idx, n_ocr=n_ocr_frames, reads=n_ocr_reads,
                              n_num=len(number_dets), n_val=len(validated_so_far))

            players = {}
            validated = number_validator.validated_dict()
            for tid, num in validated.items():
                players[str(tid)] = str(num)

            elapsed = time.monotonic() - t_start
            _log_info("ocr.complete ocr_frames={n_ocr} reads={reads} players_found={n_players} elapsed_s={elapsed:.1f}",
                       n_ocr=n_ocr_frames, reads=n_ocr_reads, n_players=len(players), elapsed=elapsed)
            if _logfire:
                logfire.info("ocr.players_identified players={players}", players=players)

            output = {
                "_meta": {"stage": "jerseys", "config_key": cfg_key, "upstream": {"tracks": track_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
                "players": players,
            }

            atomic_write_json(out, output)
            (out.parent / "_progress.json").unlink(missing_ok=True)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("OCR failed")
        if _logfire:
            logfire.error("ocr.failed error={error}", error=str(e))
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── Tracking (ByteTrack) ──────────────────────────────────────────────

@app.post("/api/track", response_model=InferenceResponse)
async def track(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        det_config_key = req.upstream_configs.get("detections", "")
        cfg_params = {"tracker": "bytetrack", "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "tracks", cfg_key, stem)

        if out.exists():
            _log_info("track.cache_hit video_id={video_id}", video_id=req.video_id)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        if not det_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Detections not found")
        det_data = json.loads(det_path.read_text())

        span = logfire.span("track.process", video_id=req.video_id, tracker="bytetrack", det_config_key=det_config_key) if _logfire else None
        if span:
            span.__enter__()

        try:
            from basket_tube.inference.vision.tracker import PlayerTracker

            fps = det_data.get("video_info", {}).get("fps", 30)
            tracker = PlayerTracker(frame_rate=fps)
            _log_info("track.initialized tracker=bytetrack fps={fps}", fps=fps)

            frames_data = []
            all_tracker_ids = set()
            track_total_frames = len(det_data["frames"])
            t_start = time.monotonic()

            for frame_det in det_data["frames"]:
                idx = frame_det["frame_index"]
                xyxy = np.array(frame_det["xyxy"])
                class_ids = np.array(frame_det["class_id"])
                confidences = np.array(frame_det["confidence"])

                # Filter to player classes only
                player_mask = np.isin(class_ids, PLAYER_CLASS_IDS)
                if not player_mask.any():
                    frames_data.append({
                        "frame_index": idx, "tracker_ids": [], "xyxy": [], "mask_rle": [],
                    })
                    continue

                player_dets = sv.Detections(
                    xyxy=xyxy[player_mask],
                    class_id=class_ids[player_mask],
                    confidence=confidences[player_mask],
                )

                tracked = tracker.update(player_dets)

                tids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
                all_tracker_ids.update(tids)

                frames_data.append({
                    "frame_index": idx,
                    "tracker_ids": tids,
                    "xyxy": tracked.xyxy.tolist(),
                    "mask_rle": [],
                })

                if idx % 10 == 0:
                    write_progress(out.parent, frame=idx, total_frames=track_total_frames)

                if idx > 0 and idx % LOG_INTERVAL == 0:
                    _log_info("track.progress frame={frame} active_tracks={n_active} total_ids={n_total}",
                              frame=idx, n_active=len(tids), n_total=len(all_tracker_ids))

            elapsed = time.monotonic() - t_start
            n_tracks = len(all_tracker_ids)
            _log_info("track.complete frames={n_frames} tracks={n_tracks} elapsed_s={elapsed:.1f} fps={fps:.1f}",
                       n_frames=len(frames_data), n_tracks=n_tracks, elapsed=elapsed,
                       fps=len(frames_data) / elapsed if elapsed > 0 else 0)

            output = {
                "_meta": {"stage": "tracks", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
                "n_frames": len(frames_data),
                "n_tracks": n_tracks,
                "frames": frames_data,
            }

            atomic_write_json(out, output)
            (out.parent / "_progress.json").unlink(missing_ok=True)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("Tracking failed")
        if _logfire:
            logfire.error("track.failed error={error}", error=str(e))
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

        span = logfire.span("classify_teams.process", video_id=req.video_id, stride=stride) if _logfire else None
        if span:
            span.__enter__()

        try:
            from basket_tube.inference.vision.classifier import extract_player_crops
            from sports import TeamClassifier

            crops = []
            crop_metadata = []
            frame_generator = sv.get_video_frames_generator(str(video_path), stride=stride)
            teams_total_frames = len(det_data["frames"])
            t_start = time.monotonic()

            for frame_idx_sampled, frame in enumerate(frame_generator):
                actual_idx = frame_idx_sampled * stride
                if actual_idx >= teams_total_frames:
                    break

                if frame_idx_sampled % 10 == 0:
                    write_progress(out.parent, frame=actual_idx, total_frames=teams_total_frames)

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

            _log_info("classify_teams.crops_collected n_crops={n} from_frames={n_frames}",
                       n=len(crops), n_frames=frame_idx_sampled + 1 if 'frame_idx_sampled' in dir() else 0)

            if len(crops) < 2:
                _log_info("classify_teams.insufficient_crops n_crops={n}", n=len(crops))
                output = {
                    "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}},
                    "palette": {"0": {"name": "Team A", "color": "#006BB6"}, "1": {"name": "Team B", "color": "#007A33"}},
                    "assignments": [],
                }
                atomic_write_json(out, output)
                (out.parent / "_progress.json").unlink(missing_ok=True)
                return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

            _log_info("classify_teams.fitting n_crops={n}", n=len(crops))
            classifier = TeamClassifier(device="cuda")
            classifier.fit(crops)
            teams = classifier.predict(crops)

            team_counts = Counter(int(t) for t in teams)
            _log_info("classify_teams.clustered team_0={t0} team_1={t1}",
                       t0=team_counts.get(0, 0), t1=team_counts.get(1, 0))

            assignments = [
                {"frame_index": meta["frame_index"], "detection_index": meta["detection_index"], "team_id": int(team_id)}
                for meta, team_id in zip(crop_metadata, teams)
            ]

            elapsed = time.monotonic() - t_start
            _log_info("classify_teams.complete assignments={n} elapsed_s={elapsed:.1f}",
                       n=len(assignments), elapsed=elapsed)

            output = {
                "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
                "palette": {"0": {"name": "Team A", "color": "#006BB6"}, "1": {"name": "Team B", "color": "#007A33"}},
                "assignments": assignments,
            }

            atomic_write_json(out, output)
            (out.parent / "_progress.json").unlink(missing_ok=True)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("Team classification failed")
        if _logfire:
            logfire.error("classify_teams.failed error={error}", error=str(e))
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


# ── Render (annotated video) ──────────────────────────────────────────

PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player, possession, jump-shot, layup-dunk, shot-block

@app.post("/api/render", response_model=InferenceResponse)
async def render(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        det_config_key = req.upstream_configs.get("detections", "")
        track_config_key = req.upstream_configs.get("tracks", "")
        teams_config_key = req.upstream_configs.get("teams", "")
        jerseys_config_key = req.upstream_configs.get("jerseys", "")

        cfg_params = {
            "det_config_key": det_config_key,
            "track_config_key": track_config_key,
            "teams_config_key": teams_config_key,
            "jerseys_config_key": jerseys_config_key,
        }
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "renders", cfg_key, stem)

        if out.exists():
            _log_info("render.cache_hit video_id={video_id}", video_id=req.video_id)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        # Load upstream artifacts
        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        track_path = artifact_path(DATA_DIR, "tracks", track_config_key, stem)
        teams_path = artifact_path(DATA_DIR, "teams", teams_config_key, stem)
        jerseys_path = artifact_path(DATA_DIR, "jerseys", jerseys_config_key, stem)

        for p, name in [(det_path, "detections"), (track_path, "tracks"), (teams_path, "teams"), (jerseys_path, "jerseys")]:
            if not p.exists():
                return InferenceResponse(status="error", config_key=cfg_key, output_path="", error=f"{name} artifact not found")

        det_data = json.loads(det_path.read_text())
        track_data = json.loads(track_path.read_text())
        teams_data = json.loads(teams_path.read_text())
        jerseys_data = json.loads(jerseys_path.read_text())

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        if not video_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error=f"Video not found: {video_path}")

        start_t = time.time()
        span = logfire.span("render.process", video_id=req.video_id) if _logfire else None
        if span:
            span.__enter__()

        try:
            video_info = sv.VideoInfo.from_video_path(str(video_path))
            total_frames = video_info.total_frames

            # --- Build lookup tables from artifacts ---

            # Jersey map: str(tracker_id) -> str(number)
            jersey_map = jerseys_data.get("players", {})

            # Team assignments: (frame_index, detection_index) -> team_id
            # Also build tracker_id -> team_id by majority vote
            team_by_frame_det = {}
            tracker_team_votes: dict[int, list[int]] = {}
            for a in teams_data.get("assignments", []):
                fi = a.get("frame_index", -1)
                di = a.get("detection_index", -1)
                tid_val = a.get("team_id", 0)
                team_by_frame_det[(fi, di)] = tid_val

            # Map detection_index to tracker_id using det + track frames
            det_frames = {f["frame_index"]: f for f in det_data.get("frames", [])}
            track_frames = {f["frame_index"]: f for f in track_data.get("frames", [])}

            # Build tracker_id -> team_id via majority vote across all assigned frames
            for a in teams_data.get("assignments", []):
                fi = a.get("frame_index", -1)
                di = a.get("detection_index", -1)
                team_id = a.get("team_id", 0)

                # Get tracker_ids from track frame, match by detection_index
                tf = track_frames.get(fi)
                if tf is None:
                    continue
                tids = tf.get("tracker_ids", [])
                if di < len(tids):
                    tid = tids[di]
                    tracker_team_votes.setdefault(tid, []).append(team_id)

            # Majority vote for each tracker
            tracker_team_map: dict[int, int] = {}
            for tid, votes in tracker_team_votes.items():
                tracker_team_map[tid] = max(set(votes), key=votes.count)

            # Team colors from palette
            palette = teams_data.get("palette", {})
            color_0 = palette.get("0", {}).get("color", "#006BB6")
            color_1 = palette.get("1", {}).get("color", "#007A33")
            team_colors = sv.ColorPalette.from_hex([color_0, color_1])

            # Annotators
            team_box_annotator = sv.BoxAnnotator(
                color=team_colors, thickness=2, color_lookup=sv.ColorLookup.INDEX
            )
            team_label_annotator = sv.LabelAnnotator(
                color=team_colors,
                text_color=sv.Color.WHITE,
                text_position=sv.Position.BOTTOM_CENTER,
                color_lookup=sv.ColorLookup.INDEX,
            )

            # Temp output path then compress
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp_video = out.parent / f"{stem}_raw.mp4"

            _log_info("render.start video_id={video_id} frames={n} trackers_with_team={nt} trackers_with_jersey={nj}",
                       video_id=req.video_id, n=total_frames,
                       nt=len(tracker_team_map), nj=len(jersey_map))

            with sv.VideoSink(str(tmp_video), video_info) as sink:
                for idx, frame in enumerate(sv.get_video_frames_generator(str(video_path))):
                    if idx % 10 == 0:
                        write_progress(out.parent, frame=idx, total_frames=total_frames)

                    track_frame = track_frames.get(idx)
                    if track_frame is None:
                        sink.write_frame(frame)
                        continue

                    # Track data: tracker_ids + xyxy
                    xyxy_list = track_frame.get("xyxy", [])
                    tracker_ids = track_frame.get("tracker_ids", [])

                    if not xyxy_list or not tracker_ids:
                        sink.write_frame(frame)
                        continue

                    # Use detection class_ids to filter to players
                    det_frame = det_frames.get(idx)
                    if det_frame is not None:
                        det_class_ids = det_frame.get("class_id", [])
                    else:
                        det_class_ids = []

                    xyxy = np.array(xyxy_list, dtype=np.float32)
                    tracker_id = np.array(tracker_ids, dtype=int)

                    # Filter: keep only entries that have tracker_ids (tracks may have fewer entries than detections)
                    n = min(len(xyxy), len(tracker_id))
                    xyxy = xyxy[:n]
                    tracker_id = tracker_id[:n]

                    if n == 0:
                        sink.write_frame(frame)
                        continue

                    detections = sv.Detections(xyxy=xyxy, tracker_id=tracker_id)

                    # Team color lookup and jersey labels per tracker
                    teams = np.array([tracker_team_map.get(int(tid), 0) for tid in tracker_id], dtype=int)
                    labels = [f"#{jersey_map.get(str(int(tid)), '?')}" for tid in tracker_id]

                    annotated = frame.copy()
                    annotated = team_box_annotator.annotate(
                        scene=annotated, detections=detections, custom_color_lookup=teams
                    )
                    annotated = team_label_annotator.annotate(
                        scene=annotated, detections=detections, labels=labels, custom_color_lookup=teams
                    )
                    sink.write_frame(annotated)

            # Compress with ffmpeg
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video),
                 "-vcodec", "libx264", "-crf", "28", str(out)],
                check=True,
            )
            tmp_video.unlink(missing_ok=True)
            (out.parent / "_progress.json").unlink(missing_ok=True)

            elapsed = time.time() - start_t
            _log_info("render.complete video_id={video_id} frames={n} elapsed={elapsed:.1f}s",
                       video_id=req.video_id, n=total_frames, elapsed=elapsed)

            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))
        finally:
            if span:
                span.__exit__(None, None, None)

    except Exception as e:
        logger.exception("Render failed")
        if _logfire:
            logfire.error("render.failed error={error}", error=str(e))
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
