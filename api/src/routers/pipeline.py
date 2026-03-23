"""Pipeline orchestrator endpoints — run, cancel, SSE events, staleness."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.src.artifacts import artifact_path, config_key
from api.src.config import settings
from api.src.schemas.pipeline import PipelineCancelResponse, PipelineRunRequest, PipelineRunResponse
from api.src.schemas.settings import AnalysisSettings
from api.src.services.pipeline_orchestrator import PipelineOrchestrator
from api.src.video_registry import resolve_stem

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_orchestrator = PipelineOrchestrator(
    gpu_url=settings.inference_gpu_url,
    data_dir=settings.data_dir,
)


def get_orchestrator() -> PipelineOrchestrator:
    return _orchestrator


@router.post("/run/{video_id}", response_model=PipelineRunResponse, status_code=202)
async def run_pipeline(video_id: str, body: PipelineRunRequest = PipelineRunRequest()):
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not found")

    try:
        _orchestrator.start_pipeline(
            video_id,
            settings=body.settings.model_dump(),
            stem=stem,
            from_stage=body.from_stage,
        )
    except RuntimeError:
        raise HTTPException(
            409,
            detail={
                "detail": "Pipeline already running for this video",
                "sse_url": f"/api/pipeline/events/{video_id}",
            },
        )

    return PipelineRunResponse(sse_url=f"/api/pipeline/events/{video_id}")


@router.post("/cancel/{video_id}", response_model=PipelineCancelResponse)
async def cancel_pipeline(video_id: str):
    cancelled = await _orchestrator.cancel_pipeline(video_id)
    return PipelineCancelResponse(cancelled_stages=cancelled)


@router.get("/events/{video_id}")
async def pipeline_events(video_id: str):
    run = _orchestrator.get_or_create_run(video_id)

    async def event_stream():
        async for event in run.bus.subscribe(cursor=0):
            data = json.dumps(event, default=str)
            event_type = event.get("event", "message")
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("pipeline_completed", "pipeline_error"):
                break

    async def stream_with_keepalive():
        """Merge SSE events with periodic keepalive comments."""
        keepalive_interval = 15
        event_iter = event_stream().__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(event_iter.__anext__(), timeout=keepalive_interval)
                yield chunk
                if chunk.startswith(": done"):
                    return
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            except StopAsyncIteration:
                return

    return StreamingResponse(
        stream_with_keepalive(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Stage → artifact dir name mapping ────────────────────────────────
_STAGE_ARTIFACT_DIR = {
    "detect": "detections",
    "track": "tracks",
    "classify-teams": "teams",
    "ocr": "jerseys",
    "court-map": "court",
}

# Stage dependency graph for cascade staleness
_STAGE_DEPS: dict[str, list[str]] = {
    "detect": [],
    "track": ["detect"],
    "classify-teams": ["detect"],
    "court-map": ["detect"],
    "ocr": ["track"],
}


def _find_existing_config_key(data_dir: Path, artifact_dir: str, stem: str) -> str | None:
    """Find the config_key of the most recent artifact for a stage."""
    stage_dir = data_dir / "analysis" / artifact_dir
    if not stage_dir.exists():
        return None
    for cfg_dir in sorted(stage_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        resolved = cfg_dir / "config.resolved.json"
        if resolved.exists():
            return cfg_dir.name
    return None


def _build_gpu_params(stage: str, stage_settings: dict, upstream_keys: dict[str, str]) -> dict:
    """Build the params dict that matches what the GPU service uses for config_key."""
    if stage == "detect":
        ds = stage_settings.get("detect", {})
        return {
            "model_id": ds.get("model_id", "basketball-player-detection-3-ycjdo/4"),
            "confidence": ds.get("confidence", 0.4),
            "iou_threshold": ds.get("iou_threshold", 0.9),
        }
    elif stage == "track":
        return {"tracker": "bytetrack", "det_config_key": upstream_keys.get("detect", "")}
    elif stage == "classify-teams":
        ts = stage_settings.get("teams", {})
        return {
            "stride": ts.get("stride", 30),
            "crop_scale": ts.get("crop_scale", 0.4),
            "det_config_key": upstream_keys.get("detect", ""),
        }
    elif stage == "court-map":
        cs = stage_settings.get("court_map", {})
        return {
            "model_id": cs.get("model_id", "basketball-court-detection-2/14"),
            "keypoint_confidence": cs.get("keypoint_confidence", 0.3),
            "anchor_confidence": cs.get("anchor_confidence", 0.5),
            "det_config_key": upstream_keys.get("detect", ""),
        }
    elif stage == "ocr":
        os_ = stage_settings.get("ocr", {})
        return {
            "model_id": os_.get("model_id", "basketball-jersey-numbers-ocr/3"),
            "n_consecutive": os_.get("n_consecutive", 3),
            "ocr_interval": os_.get("ocr_interval", 5),
            "track_config_key": upstream_keys.get("track", ""),
        }
    return {}


@router.post("/staleness/{video_id}")
async def check_staleness(video_id: str, body: PipelineRunRequest = PipelineRunRequest()):
    """Check which stages have outdated artifacts given current settings."""
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not found")

    stage_settings = body.settings.model_dump().get("stages", {})
    data_dir = settings.data_dir

    # For each stage, compute the expected config_key and compare with existing
    stage_order = ["detect", "track", "classify-teams", "court-map", "ocr"]
    expected_keys: dict[str, str] = {}
    existing_keys: dict[str, str | None] = {}
    result: dict[str, dict] = {}

    for stage in stage_order:
        artifact_dir = _STAGE_ARTIFACT_DIR[stage]

        # Build upstream keys map (using expected keys, not existing)
        upstream_map = {}
        for dep in _STAGE_DEPS[stage]:
            upstream_map[dep] = expected_keys.get(dep, "")

        params = _build_gpu_params(stage, stage_settings, upstream_map)
        expected = config_key(params)
        expected_keys[stage] = expected

        existing = _find_existing_config_key(data_dir, artifact_dir, stem)
        existing_keys[stage] = existing

        if existing is None:
            result[stage] = {"stale": False, "reason": "no artifact yet"}
        elif existing != expected:
            result[stage] = {"stale": True, "reason": f"config changed: {existing} → {expected}"}
        else:
            result[stage] = {"stale": False}

    # Cascade: if a stage is stale, all downstream are also stale
    for stage in stage_order:
        if result[stage].get("stale"):
            continue
        for dep in _STAGE_DEPS[stage]:
            if result.get(dep, {}).get("stale"):
                result[stage] = {"stale": True, "reason": f"upstream {dep} is outdated"}
                break

    return result
