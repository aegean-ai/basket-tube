"""Schemas for the pipeline orchestrator endpoints."""

from pydantic import BaseModel
from api.src.schemas.settings import AnalysisSettings


class PipelineRunRequest(BaseModel):
    settings: AnalysisSettings = AnalysisSettings()
    from_stage: str | None = None


class PipelineRunResponse(BaseModel):
    sse_url: str


class PipelineCancelResponse(BaseModel):
    cancelled_stages: list[str]


class StageEvent(BaseModel):
    """Individual SSE event payload."""
    event: str
    stage: str | None = None
    config_key: str | None = None
    timestamp: float | None = None
    progress: float | None = None
    frame: int | None = None
    total_frames: int | None = None
    duration_s: float | None = None
    skipped: bool | None = None
    error: str | None = None
    stages_completed: int | None = None
    stages_skipped: int | None = None
    stages: dict | None = None
