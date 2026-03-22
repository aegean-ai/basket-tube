"""Vision pipeline router.

Exposes 6 pipeline stage endpoints (detect, track, classify-teams, ocr,
court-map, render) plus a status endpoint.  All stage endpoints follow the
same pattern:

1. Resolve video_id → stem (404 if unknown).
2. Check upstream dependencies exist (409 if missing).
3. Compute config_key from request params.
4. Return skipped=True if output already exists.
5. Reject if already running (409).
6. Write "active" status sidecar.
7. Call GPU service.
8. On success: write "complete" status, return response.
9. On GPU error (4xx/5xx): write "error" status, raise 500.
10. On connection error: write "error" status, raise 502.
"""

import httpx
from fastapi import APIRouter, HTTPException

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

from api.src.artifacts import (
    artifact_path,
    check_stale,
    config_key,
    read_status,
    status_path_for,
    write_resolved_config,
    write_status,
)
from api.src.config import settings
from api.src.video_registry import resolve_stem
from api.src.schemas.vision import (
    ClassifyTeamsRequest,
    ClassifyTeamsResponse,
    CourtMapRequest,
    CourtMapResponse,
    DetectRequest,
    DetectResponse,
    OCRRequest,
    OCRResponse,
    PipelineStatusResponse,
    RenderRequest,
    RenderResponse,
    StageStatusResponse,
    TrackRequest,
    TrackResponse,
)
from api.src.services.vision_service import VisionService

router = APIRouter(prefix="/api/vision", tags=["vision"])

# Ordered list of pipeline stage directory names
STAGE_NAMES = ["detections", "tracks", "teams", "jerseys", "court", "renders"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vision_service() -> VisionService:
    """Create a VisionService from application settings."""
    return VisionService(
        gpu_url=settings.inference_gpu_url,
    )


def _resolve_or_404(video_id: str) -> str:
    """Resolve video_id to a title stem, raising 404 if unknown."""
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return stem


def _require_upstream(stage: str, video_id: str, stem: str, upstream_config_key: str) -> None:
    """Raise 409 if the upstream artifact for *stage* does not exist.

    Checks ``data_dir / "analysis" / stage / upstream_config_key / "{stem}.json"``.
    """
    path = artifact_path(settings.data_dir, stage, upstream_config_key, stem)
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail={"detail": f"Stage '{stage}' must be completed", "missing": [stage]},
        )


def _check_not_running(sidecar, stage_name: str) -> None:
    """Raise 409 if the stage is currently active (after crash-recovery check)."""
    status = check_stale(sidecar)
    if status.get("status") == "active":
        raise HTTPException(
            status_code=409,
            detail=f"Stage '{stage_name}' is already running",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/detect/{video_id}", response_model=DetectResponse)
async def detect(video_id: str, req: DetectRequest = DetectRequest()):
    """Run player detection on a video."""
    stem = _resolve_or_404(video_id)

    cfg_params = {
        "model_id": req.model_id,
        "confidence": req.confidence,
        "iou_threshold": req.iou_threshold,
    }
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "detections", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        return DetectResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_frames=0,
            n_detections=0,
            skipped=True,
        )

    _check_not_running(sidecar, "detect")
    write_status(sidecar, "active", config_key=cfg_key)

    svc = _get_vision_service()
    try:
        result = await svc.detect(video_id, cfg_params)
    except httpx.HTTPStatusError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"GPU service error: {exc}") from exc
    except httpx.ConnectError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"GPU service unreachable: {exc}") from exc

    write_status(sidecar, "complete", config_key=cfg_key)
    write_resolved_config(
        output_dir=out.parent,
        stage="detect",
        config_key=cfg_key,
        params=cfg_params,
        upstream={},
    )
    return DetectResponse(
        video_id=video_id,
        config_key=cfg_key,
        n_frames=result.get("n_frames", 0),
        n_detections=result.get("n_detections", 0),
        skipped=False,
    )


@router.post("/track/{video_id}", response_model=TrackResponse)
async def track(video_id: str, req: TrackRequest):
    """Run ByteTrack multi-object tracking using existing detections."""
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {
        "tracker": "bytetrack",
        "det_config_key": req.det_config_key,
    }
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "tracks", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        return TrackResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_frames=0,
            n_tracks=0,
            skipped=True,
        )

    _check_not_running(sidecar, "track")
    write_status(sidecar, "active", config_key=cfg_key)

    svc = _get_vision_service()
    try:
        result = await svc.track(video_id, cfg_params, upstream_configs={"detections": req.det_config_key})
    except httpx.HTTPStatusError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"GPU service error: {exc}") from exc
    except httpx.ConnectError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"GPU service unreachable: {exc}") from exc

    write_status(sidecar, "complete", config_key=cfg_key)
    write_resolved_config(
        output_dir=out.parent,
        stage="track",
        config_key=cfg_key,
        params=cfg_params,
        upstream={"detections": req.det_config_key},
    )
    return TrackResponse(
        video_id=video_id,
        config_key=cfg_key,
        n_frames=result.get("n_frames", 0),
        n_tracks=result.get("n_tracks", 0),
        skipped=False,
    )


@router.post("/classify-teams/{video_id}", response_model=ClassifyTeamsResponse)
async def classify_teams(video_id: str, req: ClassifyTeamsRequest):
    """Classify players into teams using colour clustering."""
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {
        "stride": req.stride,
        "crop_scale": req.crop_scale,
        "det_config_key": req.det_config_key,
    }
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "teams", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        return ClassifyTeamsResponse(
            video_id=video_id,
            config_key=cfg_key,
            palette={},
            skipped=True,
        )

    _check_not_running(sidecar, "classify-teams")
    write_status(sidecar, "active", config_key=cfg_key)

    svc = _get_vision_service()
    try:
        result = await svc.classify_teams(
            video_id, cfg_params, upstream_configs={"detections": req.det_config_key}
        )
    except httpx.HTTPStatusError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"GPU service error: {exc}") from exc
    except httpx.ConnectError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"GPU service unreachable: {exc}") from exc

    write_status(sidecar, "complete", config_key=cfg_key)
    write_resolved_config(
        output_dir=out.parent,
        stage="classify-teams",
        config_key=cfg_key,
        params=cfg_params,
        upstream={"detections": req.det_config_key},
    )
    return ClassifyTeamsResponse(
        video_id=video_id,
        config_key=cfg_key,
        palette=result.get("palette", {}),
        skipped=False,
    )


@router.post("/ocr/{video_id}", response_model=OCRResponse)
async def ocr(video_id: str, req: OCRRequest):
    """Run jersey number OCR using existing tracks."""
    stem = _resolve_or_404(video_id)
    _require_upstream("tracks", video_id, stem, req.track_config_key)

    cfg_params = {
        "model_id": req.model_id,
        "n_consecutive": req.n_consecutive,
        "ocr_interval": req.ocr_interval,
        "track_config_key": req.track_config_key,
    }
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "jerseys", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        return OCRResponse(
            video_id=video_id,
            config_key=cfg_key,
            players={},
            skipped=True,
        )

    _check_not_running(sidecar, "ocr")
    write_status(sidecar, "active", config_key=cfg_key)

    svc = _get_vision_service()
    try:
        result = await svc.ocr(
            video_id, cfg_params, upstream_configs={"tracks": req.track_config_key}
        )
    except httpx.HTTPStatusError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"GPU service error: {exc}") from exc
    except httpx.ConnectError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"GPU service unreachable: {exc}") from exc

    write_status(sidecar, "complete", config_key=cfg_key)
    write_resolved_config(
        output_dir=out.parent,
        stage="ocr",
        config_key=cfg_key,
        params=cfg_params,
        upstream={"tracks": req.track_config_key},
    )
    return OCRResponse(
        video_id=video_id,
        config_key=cfg_key,
        players=result.get("players", {}),
        skipped=False,
    )


@router.post("/court-map/{video_id}", response_model=CourtMapResponse)
async def court_map(video_id: str, req: CourtMapRequest):
    """Run court keypoint detection and homography mapping."""
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {
        "model_id": req.model_id,
        "keypoint_confidence": req.keypoint_confidence,
        "anchor_confidence": req.anchor_confidence,
        "det_config_key": req.det_config_key,
    }
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "court", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        return CourtMapResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_frames_mapped=0,
            skipped=True,
        )

    _check_not_running(sidecar, "court-map")
    write_status(sidecar, "active", config_key=cfg_key)

    svc = _get_vision_service()
    try:
        result = await svc.keypoints(
            video_id, cfg_params, upstream_configs={"detections": req.det_config_key}
        )
    except httpx.HTTPStatusError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"GPU service error: {exc}") from exc
    except httpx.ConnectError as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"GPU service unreachable: {exc}") from exc

    write_status(sidecar, "complete", config_key=cfg_key)
    write_resolved_config(
        output_dir=out.parent,
        stage="court-map",
        config_key=cfg_key,
        params=cfg_params,
        upstream={"detections": req.det_config_key},
    )
    return CourtMapResponse(
        video_id=video_id,
        config_key=cfg_key,
        n_frames_mapped=result.get("n_frames_mapped", 0),
        skipped=False,
    )


@router.post("/render/{video_id}", response_model=RenderResponse)
async def render(video_id: str, req: RenderRequest):
    """Composite all pipeline outputs into a final annotated video (stub)."""
    stem = _resolve_or_404(video_id)

    # Check all upstream stages exist
    _require_upstream("detections", video_id, stem, req.det_config_key)
    _require_upstream("tracks", video_id, stem, req.track_config_key)
    _require_upstream("teams", video_id, stem, req.teams_config_key)
    _require_upstream("jerseys", video_id, stem, req.jerseys_config_key)
    _require_upstream("court", video_id, stem, req.court_config_key)

    raise HTTPException(status_code=501, detail="Render not yet implemented")


@router.get("/status/{video_id}", response_model=PipelineStatusResponse)
async def status(video_id: str, config_key_filter: str | None = None):
    """Return the current status of all pipeline stages for a video."""
    stem = _resolve_or_404(video_id)

    stages: dict[str, StageStatusResponse] = {}

    for stage in STAGE_NAMES:
        stage_dir = settings.data_dir / "analysis" / stage
        best_sidecar = None

        if stage_dir.exists():
            # Iterate over all config_key sub-dirs for this stage
            for cfg_dir in sorted(stage_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                if config_key_filter and cfg_dir.name != config_key_filter:
                    continue
                # Look for a sidecar matching this stem
                ext = "mp4" if stage == "renders" else "json"
                artifact = cfg_dir / f"{stem}.{ext}"
                sidecar = status_path_for(artifact)
                if sidecar.exists():
                    best_sidecar = sidecar
                    break  # Use the first matching sidecar found

        if best_sidecar is not None:
            raw = check_stale(best_sidecar)
        else:
            raw = {"status": "pending"}

        stages[stage] = StageStatusResponse(
            status=raw.get("status", "pending"),
            config_key=raw.get("config_key"),
            started_at=str(raw["started_at"]) if "started_at" in raw else None,
            completed_at=str(raw["completed_at"]) if "completed_at" in raw else None,
            duration_ms=int(raw["duration_ms"]) if raw.get("duration_ms") is not None else None,
            error=raw.get("error"),
        )

    return PipelineStatusResponse(video_id=video_id, stages=stages)


@router.get("/artifacts/{stage}/{video_id}")
async def get_artifact(stage: str, video_id: str, config_key: str):
    """Return raw artifact JSON for client-side data assembly."""
    import json as json_mod
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")
    if stage not in STAGE_NAMES:
        raise HTTPException(404, f"Unknown stage '{stage}'")
    path = artifact_path(settings.data_dir, stage, config_key, stem)
    if not path.exists():
        raise HTTPException(404, f"Artifact not found: {stage}/{config_key}")
    return json_mod.loads(path.read_text())
