"""Pipeline orchestrator endpoints — run, cancel, SSE events."""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.src.config import settings
from api.src.schemas.pipeline import PipelineCancelResponse, PipelineRunRequest, PipelineRunResponse
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
