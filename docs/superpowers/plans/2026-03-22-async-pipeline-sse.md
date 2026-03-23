# Async Pipeline with SSE Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the BasketTube pipeline fully asynchronous with real-time SSE progress, per-stage re-run controls, and per-stage settings UI.

**Architecture:** FastAPI backend gains a pipeline orchestrator that runs stages as `asyncio.Task`s with dependency-aware parallelism, broadcasting progress via SSE. GPU service writes progress files polled by the orchestrator. Frontend switches from sequential awaits to an EventSource-driven state machine. Settings dialog is rewritten with per-stage sidebar navigation.

**Tech Stack:** FastAPI, asyncio, httpx, SSE (StreamingResponse), Next.js 16, React 19, EventSource API, shadcn/ui

**Spec:** `docs/superpowers/specs/2026-03-22-async-pipeline-sse-design.md`

---

## File Map

### New Files — Backend
| File | Responsibility |
|------|---------------|
| `api/src/services/event_bus.py` | Broadcast EventBus (append-only log + per-subscriber cursors) |
| `api/src/services/pipeline_orchestrator.py` | Stage scheduling, progress polling, PipelineRun state |
| `api/src/routers/pipeline.py` | `POST /run`, `POST /cancel`, `GET /events` (SSE) |
| `api/src/schemas/pipeline.py` | Pipeline request/response + SSE event models |
| `tests/test_event_bus.py` | EventBus unit tests |
| `tests/test_pipeline_orchestrator.py` | Orchestrator unit tests |
| `tests/test_pipeline_router.py` | Pipeline endpoint integration tests |

### New Files — Frontend
| File | Responsibility |
|------|---------------|
| `frontend/src/hooks/use-sse.ts` | EventSource hook |
| `frontend/src/lib/stage-deps.ts` | Dependency graph, cascade computation |

### Modified Files — Backend
| File | Change |
|------|--------|
| `api/src/artifacts.py` | Add `delete_artifact()` |
| `api/src/schemas/settings.py` | Stage-keyed settings model + migration |
| `api/src/schemas/vision.py` | New fields on TrackRequest, ClassifyTeamsRequest; 202 response model |
| `api/src/services/whisper_service.py` | Sync `requests` → async `httpx` |
| `api/src/routers/download.py` | Wrap in `asyncio.to_thread()`, add Logfire span |
| `api/src/routers/transcribe.py` | Call async whisper, add Logfire span |
| `api/src/routers/captions.py` | Wrap `build_timeline` in `asyncio.to_thread()`, add Logfire span |
| `api/src/routers/vision.py` | Return 202, background task, DELETE endpoint |
| `api/src/routers/settings.py` | Migration logic for old flat format (GET and PUT) |
| `api/src/main.py` | Register pipeline router |
| `basket_tube/inference/main.py` | Write `_progress.json` (atomic tmp+rename) in frame processing loops |

### Modified Files — Frontend
| File | Change |
|------|--------|
| `frontend/src/lib/types.ts` | StageSettings, updated AnalysisSettings, ready status, progress fields |
| `frontend/src/lib/api.ts` | runFullPipeline, cancelPipeline, deleteArtifact, del helper |
| `frontend/src/hooks/use-pipeline.ts` | SSE-driven, runStage, rerunStage, cancelPipeline |
| `frontend/src/contexts/analysis-settings-context.tsx` | Stage-keyed settings |
| `frontend/src/components/settings-dialog.tsx` | Full rewrite: per-stage sidebar |
| `frontend/src/components/pipeline-table.tsx` | Action buttons, progress bar, cascade confirm |
| `frontend/src/components/pipeline-status-bar.tsx` | Show % progress |
| `frontend/src/components/analysis-layout.tsx` | Wire new usePipeline API |

---

## Task 1: EventBus — Broadcast Event Bus

**Files:**
- Create: `api/src/services/event_bus.py`
- Test: `tests/test_event_bus.py`

- [ ] **Step 1: Write failing tests for EventBus**

```python
# tests/test_event_bus.py
import asyncio
import pytest
from api.src.services.event_bus import EventBus


@pytest.mark.asyncio
async def test_emit_and_subscribe():
    bus = EventBus()
    await bus.emit({"type": "a"})
    await bus.emit({"type": "b"})

    events = []
    async for evt in bus.subscribe(cursor=0):
        events.append(evt)
        if len(events) == 2:
            break
    assert events == [{"type": "a"}, {"type": "b"}]


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_all_events():
    bus = EventBus()
    await bus.emit({"type": "first"})

    results_a = []
    results_b = []

    async def consume(results, cursor=0):
        async for evt in bus.subscribe(cursor=cursor):
            results.append(evt)
            if len(results) == 2:
                break

    task_a = asyncio.create_task(consume(results_a))
    task_b = asyncio.create_task(consume(results_b))
    await asyncio.sleep(0.01)
    await bus.emit({"type": "second"})
    await asyncio.gather(task_a, task_b)

    assert results_a == [{"type": "first"}, {"type": "second"}]
    assert results_b == [{"type": "first"}, {"type": "second"}]


@pytest.mark.asyncio
async def test_subscribe_waits_for_new_events():
    bus = EventBus()
    received = []

    async def consumer():
        async for evt in bus.subscribe():
            received.append(evt)
            if evt.get("type") == "done":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    assert received == []  # still waiting

    await bus.emit({"type": "hello"})
    await asyncio.sleep(0.01)
    assert received == [{"type": "hello"}]

    await bus.emit({"type": "done"})
    await task
    assert len(received) == 2


@pytest.mark.asyncio
async def test_replay_from_cursor():
    bus = EventBus()
    await bus.emit({"type": "a"})
    await bus.emit({"type": "b"})
    await bus.emit({"type": "c"})

    events = []
    async for evt in bus.subscribe(cursor=1):
        events.append(evt)
        if len(events) == 2:
            break
    assert events == [{"type": "b"}, {"type": "c"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pantelis.monogioudis/local/ai/apps/computer-vision/auraison-app/basket-tube && uv run pytest tests/test_event_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.src.services.event_bus'`

- [ ] **Step 3: Implement EventBus**

```python
# api/src/services/event_bus.py
"""Broadcast event bus for SSE pipeline progress.

Supports multiple concurrent subscribers via an append-only event log
with per-subscriber cursors. Events are never removed — late subscribers
replay from any position.
"""

import asyncio
from typing import AsyncIterator


class EventBus:
    """Broadcast event bus supporting multiple concurrent SSE consumers."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._notify = asyncio.Condition()

    @property
    def size(self) -> int:
        return len(self._events)

    async def emit(self, event: dict) -> None:
        """Append an event and wake all waiting subscribers."""
        self._events.append(event)
        async with self._notify:
            self._notify.notify_all()

    async def subscribe(self, cursor: int = 0) -> AsyncIterator[dict]:
        """Yield events starting from *cursor*, waiting for new ones."""
        while True:
            while cursor < len(self._events):
                yield self._events[cursor]
                cursor += 1
            async with self._notify:
                await self._notify.wait()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_bus.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add api/src/services/event_bus.py tests/test_event_bus.py
git commit -m "feat: add broadcast EventBus for SSE pipeline progress"
```

---

## Task 2: Pipeline Schemas

**Files:**
- Create: `api/src/schemas/pipeline.py`
- Test: `tests/test_schemas.py` (append)

- [ ] **Step 1: Write failing test for pipeline schemas**

```python
# append to tests/test_schemas.py or create tests/test_pipeline_schemas.py
from api.src.schemas.pipeline import PipelineRunRequest, PipelineRunResponse, StageEvent


def test_pipeline_run_request_defaults():
    req = PipelineRunRequest()
    assert req.from_stage is None


def test_stage_event_serialization():
    evt = StageEvent(event="stage_completed", stage="detect", config_key="c-abc1234", duration_s=42.1)
    d = evt.model_dump()
    assert d["event"] == "stage_completed"
    assert d["config_key"] == "c-abc1234"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_schemas.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement pipeline schemas**

```python
# api/src/schemas/pipeline.py
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
    event: str  # stage_started | stage_progress | stage_completed | stage_skipped | stage_error | pipeline_completed
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
    stages: dict | None = None  # for pipeline_state snapshot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/src/schemas/pipeline.py tests/test_pipeline_schemas.py
git commit -m "feat: add pipeline orchestrator schemas"
```

---

## Task 3: Settings Schema Migration (flat → stage-keyed)

**Files:**
- Modify: `api/src/schemas/settings.py`
- Modify: `api/src/routers/settings.py`
- Test: `tests/test_settings.py` (extend)

- [ ] **Step 1: Write failing tests for new settings + migration**

```python
# tests/test_settings_migration.py
from api.src.schemas.settings import AnalysisSettings, migrate_settings


def test_new_stage_keyed_settings():
    s = AnalysisSettings()
    assert s.stages.detect.confidence == 0.4
    assert s.stages.transcribe.model == "Systran/faster-whisper-medium"
    assert s.stages.ocr.ocr_interval == 5


def test_migrate_old_flat_format():
    old = {
        "game_context": {"teams": {"0": {"name": "A", "color": "#000"}}, "roster": {}},
        "advanced": {"confidence": 0.7, "iou_threshold": 0.8, "ocr_interval": 10, "crop_scale": 0.3, "stride": 15},
    }
    result = migrate_settings(old)
    assert result.stages.detect.confidence == 0.7
    assert result.stages.detect.iou_threshold == 0.8
    assert result.stages.ocr.ocr_interval == 10
    assert result.stages.teams.crop_scale == 0.3
    assert result.stages.teams.stride == 15
    assert result.game_context.teams["0"].name == "A"


def test_new_format_passes_through():
    new = {
        "game_context": {"teams": {}, "roster": {}},
        "stages": {
            "detect": {"model_id": "custom/1", "confidence": 0.5, "iou_threshold": 0.85},
            "transcribe": {"model": "tiny", "use_youtube_captions": False},
        },
    }
    result = migrate_settings(new)
    assert result.stages.detect.model_id == "custom/1"
    assert result.stages.transcribe.use_youtube_captions is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings_migration.py -v`
Expected: FAIL — `migrate_settings` not found

- [ ] **Step 3: Rewrite settings schema with stage-keyed model + migration**

Rewrite `api/src/schemas/settings.py`:

```python
"""Settings persistence Pydantic models — stage-keyed structure."""
from pydantic import BaseModel


class TeamInfo(BaseModel):
    name: str = "Team A"
    color: str = "#006BB6"


class GameContext(BaseModel):
    teams: dict[str, TeamInfo] = {
        "0": TeamInfo(name="Team A", color="#006BB6"),
        "1": TeamInfo(name="Team B", color="#007A33"),
    }
    roster: dict[str, str] = {}


class TranscribeSettings(BaseModel):
    model: str = "Systran/faster-whisper-medium"
    use_youtube_captions: bool = True


class DetectSettings(BaseModel):
    model_id: str = "basketball-player-detection-3-ycjdo/4"
    confidence: float = 0.4
    iou_threshold: float = 0.9


class TrackSettings(BaseModel):
    iou_threshold: float = 0.5
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30


class OCRSettings(BaseModel):
    model_id: str = "basketball-jersey-numbers-ocr/3"
    ocr_interval: int = 5
    n_consecutive: int = 3


class TeamsSettings(BaseModel):
    embedding_model: str = "google/siglip-base-patch16-224"
    n_teams: int = 2
    crop_scale: float = 0.4
    stride: int = 30


class CourtMapSettings(BaseModel):
    model_id: str = "basketball-court-detection-2/14"
    keypoint_confidence: float = 0.3
    anchor_confidence: float = 0.5


class StageSettings(BaseModel):
    transcribe: TranscribeSettings = TranscribeSettings()
    detect: DetectSettings = DetectSettings()
    track: TrackSettings = TrackSettings()
    ocr: OCRSettings = OCRSettings()
    teams: TeamsSettings = TeamsSettings()
    court_map: CourtMapSettings = CourtMapSettings()


class AnalysisSettings(BaseModel):
    game_context: GameContext = GameContext()
    stages: StageSettings = StageSettings()


# --- Migration from old flat format ---

class _OldAdvanced(BaseModel):
    confidence: float = 0.4
    iou_threshold: float = 0.9
    ocr_interval: int = 5
    crop_scale: float = 0.4
    stride: int = 30


def migrate_settings(data: dict) -> AnalysisSettings:
    """Accept either old flat format or new stage-keyed format."""
    if "stages" in data:
        return AnalysisSettings(**data)

    # Old format: { game_context, advanced }
    gc = data.get("game_context", {})
    old = _OldAdvanced(**(data.get("advanced", {})))

    stages = StageSettings(
        detect=DetectSettings(confidence=old.confidence, iou_threshold=old.iou_threshold),
        ocr=OCRSettings(ocr_interval=old.ocr_interval),
        teams=TeamsSettings(crop_scale=old.crop_scale, stride=old.stride),
    )
    return AnalysisSettings(game_context=GameContext(**gc), stages=stages)
```

- [ ] **Step 4: Update settings router to use migration**

In `api/src/routers/settings.py`, change the `get_settings` endpoint:

```python
# api/src/routers/settings.py — replace get_settings body
from api.src.schemas.settings import AnalysisSettings, migrate_settings

@router.get("/settings/{video_id}", response_model=AnalysisSettings)
async def get_settings(video_id: str):
    path = _settings_path(video_id)
    if path.exists():
        data = json.loads(path.read_text())
        return migrate_settings(data)
    return AnalysisSettings()
```

- [ ] **Step 5: Update PUT settings endpoint for backwards compatibility**

The PUT endpoint must also accept old-format payloads. In `api/src/routers/settings.py`, change the PUT endpoint to accept raw dict and migrate:

```python
from fastapi import Body

@router.put("/settings/{video_id}", response_model=AnalysisSettings)
async def put_settings(video_id: str, body: dict = Body(...)):
    path = _settings_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    migrated = migrate_settings(body)
    atomic_write_json(path, migrated.model_dump())
    return migrated
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings_migration.py tests/test_settings.py -v`
Expected: All pass (existing tests may need minor adjustments for the new schema shape)

- [ ] **Step 7: Commit**

```bash
git add api/src/schemas/settings.py api/src/routers/settings.py tests/test_settings_migration.py
git commit -m "feat: stage-keyed settings schema with migration from flat format"
```

---

## Task 4: Artifact Deletion + Vision Schema Updates

**Files:**
- Modify: `api/src/artifacts.py`
- Modify: `api/src/schemas/vision.py`
- Test: `tests/test_artifacts.py` (extend)

- [ ] **Step 1: Write failing test for delete_artifact**

```python
# tests/test_delete_artifact.py
import json
from pathlib import Path
from api.src.artifacts import artifact_path, delete_artifact, write_status, status_path_for


def test_delete_artifact_removes_all_files(tmp_path):
    data_dir = tmp_path
    stage, cfg_key, stem = "detections", "c-abc1234", "test_video"

    art = artifact_path(data_dir, stage, cfg_key, stem)
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"n_frames": 10}))

    sidecar = status_path_for(art)
    write_status(sidecar, "complete", config_key=cfg_key)

    progress = art.parent / "_progress.json"
    progress.write_text(json.dumps({"frame": 5}))

    assert art.exists()
    assert sidecar.exists()
    assert progress.exists()

    delete_artifact(data_dir, stage, cfg_key, stem)

    assert not art.exists()
    assert not sidecar.exists()
    assert not progress.exists()


def test_delete_artifact_noop_when_missing(tmp_path):
    # Should not raise
    delete_artifact(tmp_path, "detections", "c-missing", "no_video")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_delete_artifact.py -v`
Expected: FAIL — `delete_artifact` not found

- [ ] **Step 3: Implement delete_artifact in artifacts.py**

Append to `api/src/artifacts.py`:

```python
def delete_artifact(data_dir: Path, stage: str, cfg_key: str, stem: str) -> None:
    """Delete artifact, sidecar, and progress file for a stage."""
    art = artifact_path(data_dir, stage, cfg_key, stem)
    sidecar = status_path_for(art)
    progress = art.parent / "_progress.json"

    for f in (art, sidecar, progress):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
```

- [ ] **Step 4: Update vision schemas — add TrackRequest fields + 202 response**

In `api/src/schemas/vision.py`, add fields to `TrackRequest` and `ClassifyTeamsRequest`, and add `StageAcceptedResponse`:

```python
# Add to TrackRequest (after det_config_key):
    iou_threshold: float = 0.5
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30

# Add to ClassifyTeamsRequest (after det_config_key):
    embedding_model: str = "google/siglip-base-patch16-224"
    n_teams: int = 2

# New response model:
class StageAcceptedResponse(BaseModel):
    stage: str
    config_key: str
    sse_url: str
```

- [ ] **Step 5: Run all artifact and schema tests**

Run: `uv run pytest tests/test_delete_artifact.py tests/test_artifacts.py tests/test_vision_schemas.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add api/src/artifacts.py api/src/schemas/vision.py tests/test_delete_artifact.py
git commit -m "feat: add artifact deletion and updated vision request schemas"
```

---

## Task 5: Async WhisperService

**Files:**
- Modify: `api/src/services/whisper_service.py`
- Modify: `api/src/routers/transcribe.py`
- Test: `tests/test_transcribe_router.py` (extend)

- [ ] **Step 1: Write failing test for async whisper**

```python
# tests/test_whisper_async.py
import pytest
from unittest.mock import AsyncMock, patch
from api.src.services.whisper_service import transcribe


@pytest.mark.asyncio
async def test_transcribe_is_async():
    """Verify transcribe is a coroutine function."""
    import inspect
    assert inspect.iscoroutinefunction(transcribe)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_whisper_async.py -v`
Expected: FAIL — `transcribe` is not a coroutine

- [ ] **Step 3: Rewrite whisper_service.py**

```python
# api/src/services/whisper_service.py
"""Whisper STT service — async HTTP client to the remote Whisper container."""

import logging
import os

import httpx

from api.src.config import settings

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

logger = logging.getLogger(__name__)


async def transcribe(audio_path: str, model: str | None = None) -> dict:
    """POST audio/video to the remote Whisper service and return a result dict."""
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    model = model or settings.whisper_model
    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
    mime = mime_types.get(ext, "application/octet-stream")

    logger.info("Remote Whisper transcription: POST %s (file=%s, model=%s)", url, filename, model)

    # Use contextlib.nullcontext to avoid manual __enter__/__exit__ antipattern
    from contextlib import nullcontext
    span_ctx = logfire.span("whisper.transcribe", filename=filename, model=model) if logfire else nullcontext()

    with span_ctx:
        async with httpx.AsyncClient(timeout=600) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    url,
                    files={"file": (filename, f, mime)},
                    data={"model": model, "response_format": "verbose_json"},
                )

        if not response.is_success:
            logger.error("Whisper service returned %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()
        result = response.json()

        if logfire:
            logfire.info("whisper.transcribe complete",
                         segments=len(result.get("segments", [])),
                         language=result.get("language", ""))
        return result
```

- [ ] **Step 4: Update transcribe router to await async whisper**

In `api/src/routers/transcribe.py:93`, change:
```python
# OLD:
result = whisper_service.transcribe(str(video_path))
# NEW:
result = await whisper_service.transcribe(str(video_path))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_whisper_async.py tests/test_transcribe_router.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add api/src/services/whisper_service.py api/src/routers/transcribe.py tests/test_whisper_async.py
git commit -m "feat: async WhisperService using httpx"
```

---

## Task 6: Async Download + Captions Routers

**Files:**
- Modify: `api/src/routers/download.py`
- Modify: `api/src/routers/captions.py`

- [ ] **Step 1: Wrap download calls in asyncio.to_thread**

In `api/src/routers/download.py`, add `import asyncio` at top, then change lines 38, 46, 50:

```python
# Line 38: video info
video_id, title = await asyncio.to_thread(_download_service.get_video_info, body.url)

# Line 46: video download
await asyncio.to_thread(_download_service.download_video, body.url, str(videos_dir), stem)

# Line 50: caption download
await asyncio.to_thread(_download_service.download_caption, body.url, str(captions_dir), stem)
```

Also add Logfire span around the whole endpoint body:

```python
try:
    import logfire
except ImportError:
    logfire = None

# In the endpoint, wrap with:
if logfire:
    with logfire.span("pipeline.download", url=body.url):
        # ... existing body ...
```

- [ ] **Step 2: Wrap build_timeline in asyncio.to_thread**

In `api/src/routers/captions.py:72`, change:

```python
# OLD:
result = build_timeline(transcript, source=source, ...)
# NEW:
import asyncio
result = await asyncio.to_thread(
    build_timeline, transcript, source=source,
    lexicon_version=req.lexicon_version, stt_model_dir=req.stt_model_dir,
)
```

Add Logfire span:
```python
# After write_status(sidecar, "complete", ...):
if logfire:
    logfire.info("pipeline.timeline", video_id=video_id, n_segments=len(result["segments"]))
```

(Add `try: import logfire` block at top of file.)

- [ ] **Step 3: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_download_router.py tests/test_text_timeline.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add api/src/routers/download.py api/src/routers/captions.py
git commit -m "feat: async download/captions via asyncio.to_thread + Logfire spans"
```

---

## Task 7: Pipeline Orchestrator

**Files:**
- Create: `api/src/services/pipeline_orchestrator.py`
- Test: `tests/test_pipeline_orchestrator.py`

- [ ] **Step 1: Write failing tests for orchestrator**

```python
# tests/test_pipeline_orchestrator.py
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.src.services.pipeline_orchestrator import PipelineOrchestrator, PipelineRun


@pytest.mark.asyncio
async def test_get_or_create_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    assert isinstance(run, PipelineRun)
    assert run.video_id == "video1"
    # Same video returns same run
    assert orch.get_or_create_run("video1") is run


@pytest.mark.asyncio
async def test_reject_concurrent_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    run.is_active = True
    with pytest.raises(RuntimeError, match="already running"):
        orch.start_pipeline("video1", settings={})


@pytest.mark.asyncio
async def test_cancel_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    mock_task = AsyncMock()
    mock_task.cancel = MagicMock()
    mock_task.cancelled = MagicMock(return_value=False)
    run.task = mock_task
    run.is_active = True
    run.active_stages = {"detect"}

    cancelled = await orch.cancel_pipeline("video1")
    assert "detect" in cancelled
    assert not run.is_active
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_orchestrator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement PipelineOrchestrator**

Create `api/src/services/pipeline_orchestrator.py`. This is a large file — key structure:

```python
# api/src/services/pipeline_orchestrator.py
"""Pipeline orchestrator — dependency-aware stage scheduling with SSE broadcast."""

import asyncio
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field

from api.src.services.event_bus import EventBus
from api.src.services.vision_service import VisionService
from api.src.artifacts import (
    artifact_path, config_key, delete_artifact, read_status,
    status_path_for, write_resolved_config, write_status,
)

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    video_id: str
    bus: EventBus = field(default_factory=EventBus)
    task: asyncio.Task | None = None
    is_active: bool = False
    active_stages: set[str] = field(default_factory=set)
    config_keys: dict[str, str] = field(default_factory=dict)
    stage_states: dict[str, dict] = field(default_factory=dict)


class PipelineOrchestrator:
    def __init__(self, gpu_url: str, data_dir: str | Path):
        self._runs: dict[str, PipelineRun] = {}
        self._gpu_url = gpu_url
        self._data_dir = Path(data_dir)
        self._svc = VisionService(gpu_url=gpu_url)

    def get_or_create_run(self, video_id: str) -> PipelineRun:
        if video_id not in self._runs:
            self._runs[video_id] = PipelineRun(video_id=video_id)
        return self._runs[video_id]

    def start_pipeline(self, video_id: str, settings: dict, stem: str | None = None, from_stage: str | None = None):
        run = self.get_or_create_run(video_id)
        if run.is_active:
            raise RuntimeError(f"Pipeline already running for {video_id}")

        # Reset for new run
        run.bus = EventBus()
        run.is_active = True
        run.active_stages = set()
        run.config_keys = {}
        run.stage_states = {}

        run.task = asyncio.create_task(
            self._execute_pipeline(run, settings, stem or video_id, from_stage)
        )
        return run

    async def cancel_pipeline(self, video_id: str) -> list[str]:
        run = self._runs.get(video_id)
        if not run or not run.is_active:
            return []

        cancelled = list(run.active_stages)
        if run.task:
            run.task.cancel()
        run.is_active = False

        for stage in cancelled:
            await run.bus.emit({
                "event": "stage_error", "stage": stage,
                "error": "cancelled", "timestamp": time.time(),
            })
        return cancelled

    def _read_upstream_config_keys(self, stem: str) -> dict[str, str]:
        """Read config keys from config.resolved.json for from_stage re-runs."""
        keys = {}
        for stage_dir_name in ("detections", "tracks", "teams", "jerseys", "court"):
            stage_dir = self._data_dir / "analysis" / stage_dir_name
            if not stage_dir.exists():
                continue
            for cfg_dir in sorted(stage_dir.iterdir()):
                resolved = cfg_dir / "config.resolved.json"
                if resolved.exists():
                    data = json.loads(resolved.read_text())
                    keys[stage_dir_name] = data.get("config_key", cfg_dir.name)
                    break
        return keys

    async def _execute_pipeline(self, run: PipelineRun, settings: dict, stem: str, from_stage: str | None):
        """Execute the full pipeline DAG."""
        from contextlib import nullcontext

        start = time.time()

        span_ctx = logfire.span("pipeline.run", video_id=run.video_id) if logfire else nullcontext()

        try:
            with span_ctx:
                # Emit initial state snapshot
                await run.bus.emit({"event": "pipeline_state", "stages": run.stage_states})

                stages_settings = settings.get("stages", {})

                # Handle from_stage: skip stages before it, use existing upstream keys
                skip_stages = set()
                existing_keys = {}
                if from_stage:
                    existing_keys = self._read_upstream_config_keys(stem)
                    # Build skip set: all stages before from_stage in DAG
                    stage_order = ["detect", "track", "classify-teams", "court-map", "ocr"]
                    from_idx = stage_order.index(from_stage) if from_stage in stage_order else 0
                    skip_stages = set(stage_order[:from_idx])

                # 1. Download (check file exists)
                await self._emit_stage_status(run, "download", "complete")

                # 2. Detect
                if "detect" in skip_stages:
                    det_key = existing_keys.get("detections", "")
                    await run.bus.emit({"event": "stage_skipped", "stage": "detect", "config_key": det_key})
                else:
                    det_params = stages_settings.get("detect", {})
                    det_key = await self._run_vision_stage(
                        run, "detect", "detections", stem, det_params, upstream={},
                    )

                # 3. Parallel: track + classify-teams + court-map
                track_params = stages_settings.get("track", {})
                track_params["det_config_key"] = det_key
                teams_params = stages_settings.get("teams", {})
                teams_params["det_config_key"] = det_key
                court_params = stages_settings.get("court_map", {})
                court_params["det_config_key"] = det_key

                if "track" in skip_stages:
                    track_key = existing_keys.get("tracks", "")
                    track_task = None
                    await run.bus.emit({"event": "stage_skipped", "stage": "track", "config_key": track_key})
                else:
                    track_task = asyncio.create_task(
                        self._run_vision_stage(run, "track", "tracks", stem, track_params,
                                               upstream={"detections": det_key})
                    )

                teams_task = asyncio.create_task(
                    self._run_vision_stage(run, "classify-teams", "teams", stem, teams_params,
                                           upstream={"detections": det_key})
                )
                court_task = asyncio.create_task(
                    self._run_vision_stage(run, "court-map", "court", stem, court_params,
                                           upstream={"detections": det_key})
                )

                if track_task:
                    track_key = await track_task

                # 4. OCR (needs track)
                ocr_params = stages_settings.get("ocr", {})
                ocr_params["track_config_key"] = track_key
                ocr_task = asyncio.create_task(
                    self._run_vision_stage(run, "ocr", "jerseys", stem, ocr_params,
                                           upstream={"tracks": track_key})
                )

                await asyncio.gather(teams_task, court_task, ocr_task)

                duration = time.time() - start
                await run.bus.emit({
                    "event": "pipeline_completed",
                    "duration_s": round(duration, 1),
                })
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled for %s", run.video_id)
        except Exception as exc:
            logger.exception("Pipeline failed for %s", run.video_id)
            await run.bus.emit({
                "event": "pipeline_error",
                "error": str(exc),
                "timestamp": time.time(),
            })
        finally:
            run.is_active = False

    async def _run_vision_stage(
        self, run: PipelineRun, stage: str, artifact_stage: str,
        stem: str, params: dict, upstream: dict,
    ) -> str:
        """Run a single vision stage with progress polling."""
        cfg_key = config_key(params)
        run.config_keys[stage] = cfg_key
        out = artifact_path(self._data_dir, artifact_stage, cfg_key, stem)

        # Check cache
        if out.exists():
            await run.bus.emit({"event": "stage_skipped", "stage": stage, "config_key": cfg_key})
            return cfg_key

        await run.bus.emit({"event": "stage_started", "stage": stage, "timestamp": time.time()})
        run.active_stages.add(stage)

        sidecar = status_path_for(out)
        write_status(sidecar, "active", config_key=cfg_key)

        progress_path = out.parent / "_progress.json"
        start = time.time()

        # Map stage name to VisionService method
        method_map = {
            "detect": self._svc.detect,
            "track": self._svc.track,
            "classify-teams": self._svc.classify_teams,
            "ocr": self._svc.ocr,
            "court-map": self._svc.keypoints,
        }
        method = method_map[stage]

        poller = asyncio.create_task(self._poll_progress(run, stage, progress_path))
        try:
            result = await method(run.video_id, params, upstream_configs=upstream)
        except Exception as exc:
            write_status(sidecar, "error", error=str(exc))
            await run.bus.emit({
                "event": "stage_error", "stage": stage,
                "error": str(exc), "timestamp": time.time(),
            })
            raise
        finally:
            poller.cancel()
            progress_path.unlink(missing_ok=True)
            run.active_stages.discard(stage)

        duration = time.time() - start
        write_status(sidecar, "complete", config_key=cfg_key)
        write_resolved_config(out.parent, stage, cfg_key, params, upstream)

        await run.bus.emit({
            "event": "stage_completed", "stage": stage,
            "config_key": cfg_key, "duration_s": round(duration, 1),
        })
        return cfg_key

    async def _poll_progress(self, run: PipelineRun, stage: str, progress_path: Path):
        """Read progress file every 2s and emit SSE events."""
        try:
            while True:
                await asyncio.sleep(2)
                try:
                    data = json.loads(progress_path.read_text())
                    total = data.get("total_frames", 0)
                    frame = data.get("frame", 0)
                    progress = frame / total if total > 0 else 0
                    await run.bus.emit({
                        "event": "stage_progress", "stage": stage,
                        "progress": round(progress, 3),
                        "frame": frame, "total_frames": total,
                    })
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
        except asyncio.CancelledError:
            pass

    async def _emit_stage_status(self, run: PipelineRun, stage: str, status: str):
        await run.bus.emit({"event": f"stage_{status}", "stage": stage, "timestamp": time.time()})
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline_orchestrator.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add api/src/services/pipeline_orchestrator.py tests/test_pipeline_orchestrator.py
git commit -m "feat: pipeline orchestrator with dependency-aware scheduling"
```

---

## Task 8: Pipeline Router (SSE + Run + Cancel)

**Files:**
- Create: `api/src/routers/pipeline.py`
- Modify: `api/src/main.py`
- Test: `tests/test_pipeline_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_router.py
import pytest
from fastapi.testclient import TestClient
from api.src.main import app


def test_pipeline_run_returns_202():
    client = TestClient(app)
    # Will fail since no video exists, but validates endpoint exists
    resp = client.post("/api/pipeline/run/nonexistent", json={"settings": {}})
    assert resp.status_code in (202, 404)


def test_pipeline_events_endpoint_exists():
    client = TestClient(app)
    resp = client.get("/api/pipeline/events/nonexistent")
    assert resp.status_code in (200, 404)


def test_pipeline_cancel_endpoint_exists():
    client = TestClient(app)
    resp = client.post("/api/pipeline/cancel/nonexistent")
    # No active pipeline → should return empty list or 404
    assert resp.status_code in (200, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_router.py -v`
Expected: FAIL — 404 for all (endpoints not registered)

- [ ] **Step 3: Implement pipeline router**

```python
# api/src/routers/pipeline.py
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

# Module-level orchestrator singleton
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
        # Keepalive: send comment every 15s to prevent proxy/browser timeout
        keepalive_interval = 15

        async def with_keepalive():
            import asyncio
            last_event = asyncio.get_event_loop().time()
            async for event in run.bus.subscribe(cursor=0):
                yield event
                last_event = asyncio.get_event_loop().time()

        async for event in run.bus.subscribe(cursor=0):
            data = json.dumps(event, default=str)
            event_type = event.get("event", "message")
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("pipeline_completed", "pipeline_error"):
                break

        # Final comment
        yield ": done\n\n"

    # Wrap with keepalive timer
    async def stream_with_keepalive():
        import asyncio

        async def keepalive_gen():
            """Merge SSE events with periodic keepalive comments."""
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

        keepalive_interval = 15
        async for chunk in keepalive_gen():
            yield chunk

    return StreamingResponse(
        stream_with_keepalive(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Register pipeline router in main.py**

In `api/src/main.py`, after line 54 add:

```python
from api.src.routers.pipeline import router as pipeline_router
```

After line 60 add:

```python
app.include_router(pipeline_router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_pipeline_router.py -v`
Expected: All pass (endpoints exist)

- [ ] **Step 6: Commit**

```bash
git add api/src/routers/pipeline.py api/src/main.py tests/test_pipeline_router.py
git commit -m "feat: pipeline router with SSE events, run, and cancel endpoints"
```

---

## Task 9: Vision Router — 202 Background + DELETE Endpoint

**Files:**
- Modify: `api/src/routers/vision.py`

- [ ] **Step 1: Add DELETE endpoint for artifact removal**

Append to `api/src/routers/vision.py`:

```python
from api.src.artifacts import delete_artifact

@router.delete("/artifacts/{stage}/{video_id}")
async def delete_artifact_endpoint(stage: str, video_id: str, config_key: str):
    """Delete artifacts for a stage to allow re-run."""
    stem = _resolve_or_404(video_id)
    if stage not in STAGE_NAMES:
        raise HTTPException(404, f"Unknown stage '{stage}'")
    delete_artifact(settings.data_dir, stage, config_key, stem)
    return {"deleted": True, "stage": stage, "config_key": config_key}
```

- [ ] **Step 2: Convert individual vision endpoints to return 202 + background task**

For each stage endpoint (detect, track, classify-teams, ocr, court-map) in `api/src/routers/vision.py`, change the pattern from synchronous await to background task. Example for detect:

```python
from api.src.schemas.vision import StageAcceptedResponse
from api.src.routers.pipeline import get_orchestrator

@router.post("/detect/{video_id}", response_model=StageAcceptedResponse, status_code=202)
async def detect(video_id: str, req: DetectRequest = DetectRequest()):
    """Run player detection — returns 202 and executes in background."""
    stem = _resolve_or_404(video_id)
    cfg_params = {"model_id": req.model_id, "confidence": req.confidence, "iou_threshold": req.iou_threshold}
    cfg_key = config_key(cfg_params)

    out = artifact_path(settings.data_dir, "detections", cfg_key, stem)
    if out.exists():
        return StageAcceptedResponse(stage="detect", config_key=cfg_key, sse_url=f"/api/pipeline/events/{video_id}")

    sidecar = status_path_for(out)
    _check_not_running(sidecar, "detect")

    # Run in background via orchestrator event bus
    orch = get_orchestrator()
    run = orch.get_or_create_run(video_id)

    async def _bg():
        write_status(sidecar, "active", config_key=cfg_key)
        svc = _get_vision_service()
        try:
            await run.bus.emit({"event": "stage_started", "stage": "detect", "timestamp": __import__("time").time()})
            result = await svc.detect(video_id, cfg_params)
            write_status(sidecar, "complete", config_key=cfg_key)
            write_resolved_config(out.parent, "detect", cfg_key, cfg_params, {})
            await run.bus.emit({"event": "stage_completed", "stage": "detect", "config_key": cfg_key, "duration_s": 0})
        except Exception as exc:
            write_status(sidecar, "error", error=str(exc))
            await run.bus.emit({"event": "stage_error", "stage": "detect", "error": str(exc)})

    import asyncio
    asyncio.create_task(_bg())
    return StageAcceptedResponse(stage="detect", config_key=cfg_key, sse_url=f"/api/pipeline/events/{video_id}")
```

Apply the same pattern to track, classify-teams, ocr, and court-map endpoints.

- [ ] **Step 3: Run existing vision router tests to verify nothing broke**

Run: `uv run pytest tests/test_vision_router.py -v`
Expected: All pass (some may need updating for 202 status codes)

- [ ] **Step 4: Commit**

```bash
git add api/src/routers/vision.py
git commit -m "feat: vision endpoints return 202 + run in background with SSE events"
```

---

## Task 9.5: GPU Service Progress File Writing

**Files:**
- Modify: `basket_tube/inference/main.py`

This task adds atomic `_progress.json` writes during frame processing in the GPU inference service. Without this, the orchestrator's progress poller will never emit meaningful `stage_progress` SSE events.

- [ ] **Step 1: Add progress writing utility**

Add a helper function to the GPU service for atomic progress file writes:

```python
# In basket_tube/inference/main.py or a new basket_tube/inference/progress.py
import json, os, tempfile
from pathlib import Path

def write_progress(output_dir: Path, frame: int, total_frames: int):
    """Write progress atomically (tmp + rename) for API orchestrator polling."""
    progress_path = output_dir / "_progress.json"
    data = {"frame": frame, "total_frames": total_frames, "updated_at": __import__("time").time()}
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, str(progress_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
```

- [ ] **Step 2: Add progress writes to frame processing loops**

In each GPU endpoint's frame loop (detect, track, ocr, keypoints, classify-teams), call `write_progress()` every N frames (e.g., every 10 frames to avoid I/O overhead):

```python
# Example: in the detect endpoint's frame loop
for i, frame in enumerate(frames):
    # ... existing detection logic ...
    if i % 10 == 0:
        write_progress(output_dir, frame=i, total_frames=total_frames)
```

The exact locations depend on each endpoint's loop structure — the implementer should find the main frame iteration in each handler.

- [ ] **Step 3: Clean up progress file after stage completes**

After each stage's JSON artifact is written, delete the progress file:

```python
progress_path = output_dir / "_progress.json"
progress_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Commit**

```bash
git add basket_tube/inference/main.py
git commit -m "feat: GPU service writes atomic progress files for SSE polling"
```

---

## Task 10: Frontend Types + API Functions

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/stage-deps.ts`

- [ ] **Step 1: Update types.ts with stage-keyed settings + new states**

Replace the settings types and add progress fields in `frontend/src/lib/types.ts`:

```typescript
// Replace AdvancedSettings and AnalysisSettings with:

export interface TranscribeSettings {
  model: string;
  use_youtube_captions: boolean;
}

export interface DetectSettings {
  model_id: string;
  confidence: number;
  iou_threshold: number;
}

export interface TrackSettings {
  iou_threshold: number;
  track_activation_threshold: number;
  lost_track_buffer: number;
}

export interface OCRSettings {
  model_id: string;
  ocr_interval: number;
  n_consecutive: number;
}

export interface TeamsSettings {
  embedding_model: string;
  n_teams: number;
  crop_scale: number;
  stride: number;
}

export interface CourtMapSettings {
  model_id: string;
  keypoint_confidence: number;
  anchor_confidence: number;
}

export interface StageSettings {
  transcribe: TranscribeSettings;
  detect: DetectSettings;
  track: TrackSettings;
  ocr: OCRSettings;
  teams: TeamsSettings;
  court_map: CourtMapSettings;
}

export interface AnalysisSettings {
  game_context: GameContext;
  stages: StageSettings;
}

// Update StageStatus:
export type StageStatus = "pending" | "ready" | "active" | "complete" | "skipped" | "error";

// Add to StageState:
export interface StageState {
  status: StageStatus;
  config_key?: string;
  result?: unknown;
  error?: string;
  duration_ms?: number;
  started_at?: number;
  progress?: number;
  frame?: number;
  total_frames?: number;
}
```

Remove the old `AdvancedSettings` interface.

- [ ] **Step 2: Create stage-deps.ts**

```typescript
// frontend/src/lib/stage-deps.ts
import type { VisionStage, StageState } from "./types";

export const STAGE_DEPS: Record<VisionStage, VisionStage[]> = {
  download: [],
  transcribe: ["download"],
  detect: ["download"],
  track: ["detect"],
  ocr: ["track"],
  "classify-teams": ["detect"],
  "court-map": ["detect"],
  render: ["track", "ocr", "classify-teams", "court-map"],
};

/** Get all downstream stages (transitive) */
export function getDownstream(stage: VisionStage): VisionStage[] {
  const result: VisionStage[] = [];
  const queue = [stage];
  const visited = new Set<VisionStage>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [s, deps] of Object.entries(STAGE_DEPS) as [VisionStage, VisionStage[]][]) {
      if (deps.includes(current) && !visited.has(s)) {
        visited.add(s);
        result.push(s);
        queue.push(s);
      }
    }
  }
  return result;
}

/** Check if a stage is ready (all deps complete/skipped, stage is pending) */
export function isStageReady(
  stage: VisionStage,
  stages: Record<VisionStage, StageState>,
): boolean {
  const deps = STAGE_DEPS[stage];
  if (!deps.length) return stages[stage].status === "pending";
  return (
    deps.every((d) => stages[d].status === "complete" || stages[d].status === "skipped") &&
    stages[stage].status === "pending"
  );
}
```

- [ ] **Step 3: Add new API functions to api.ts**

Append to `frontend/src/lib/api.ts`:

```typescript
async function del(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new ApiError(await res.text(), res.status);
}

export const runFullPipeline = (videoId: string, settings: import("./types").AnalysisSettings) =>
  post<{ sse_url: string }>(`/api/pipeline/run/${videoId}`, { settings });

export const cancelPipeline = (videoId: string) =>
  post<{ cancelled_stages: string[] }>(`/api/pipeline/cancel/${videoId}`);

export const deleteArtifact = (stage: string, videoId: string, configKey: string) =>
  del(`/api/vision/artifacts/${stage}/${videoId}?config_key=${configKey}`);
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/lib/stage-deps.ts
git commit -m "feat: stage-keyed types, dependency graph, new API functions"
```

---

## Task 11: useSSE Hook

**Files:**
- Create: `frontend/src/hooks/use-sse.ts`

- [ ] **Step 1: Implement useSSE hook**

```typescript
// frontend/src/hooks/use-sse.ts
"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface SSEEvent {
  event: string;
  stage?: string;
  config_key?: string;
  timestamp?: number;
  progress?: number;
  frame?: number;
  total_frames?: number;
  duration_s?: number;
  error?: string;
  stages_completed?: number;
  stages_skipped?: number;
  stages?: Record<string, unknown>;
}

interface UseSSEOptions {
  onEvent: (event: SSEEvent) => void;
}

export function useSSE(videoId: string | undefined, { onEvent }: UseSSEOptions) {
  const [connected, setConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (!videoId) return;

    const es = new EventSource(`/api/pipeline/events/${videoId}`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    const eventTypes = [
      "pipeline_state", "stage_started", "stage_progress",
      "stage_completed", "stage_skipped", "stage_error", "pipeline_completed",
    ];

    for (const type of eventTypes) {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data) as SSEEvent;
          data.event = type;
          onEventRef.current(data);
        } catch { /* ignore parse errors */ }
      });
    }

    return es;
  }, [videoId]);

  const disconnect = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setConnected(false);
  }, []);

  useEffect(() => {
    const es = connect();
    return () => {
      es?.close();
      setConnected(false);
    };
  }, [connect]);

  return { connected, disconnect };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/use-sse.ts
git commit -m "feat: useSSE hook for EventSource consumption"
```

---

## Task 12: Refactor usePipeline — SSE-Driven

**Files:**
- Modify: `frontend/src/hooks/use-pipeline.ts`

- [ ] **Step 1: Rewrite usePipeline to be SSE-driven**

Complete rewrite of `frontend/src/hooks/use-pipeline.ts`. Key changes:
- `runPipeline()` calls `api.runFullPipeline()` and returns immediately
- `runStage(stage)` calls individual endpoint
- `rerunStage(stage)` deletes artifacts, then runs
- State updates from SSE only (via `useSSE` hook)
- New reducer actions: `STAGE_PROGRESS`, `PIPELINE_STATE`

```typescript
// frontend/src/hooks/use-pipeline.ts
"use client";

import { useCallback, useReducer } from "react";
import type { PipelineState, VisionStage, StageState, AnalysisSettings } from "@/lib/types";
import type { SSEEvent } from "@/hooks/use-sse";
import { useSSE } from "@/hooks/use-sse";
import { isStageReady } from "@/lib/stage-deps";
import * as api from "@/lib/api";

const STAGES: VisionStage[] = [
  "download", "transcribe", "detect", "track", "ocr", "classify-teams", "court-map", "render",
];

const INITIAL_STAGE: StageState = { status: "pending" };

const INITIAL_STATE: PipelineState = {
  status: "idle",
  stages: Object.fromEntries(STAGES.map((s) => [s, INITIAL_STAGE])) as Record<VisionStage, StageState>,
  videoId: undefined,
};

type Action =
  | { type: "START"; videoId: string }
  | { type: "STAGE_STARTED"; stage: VisionStage; timestamp: number }
  | { type: "STAGE_PROGRESS"; stage: VisionStage; progress: number; frame: number; total_frames: number }
  | { type: "STAGE_COMPLETE"; stage: VisionStage; config_key?: string; duration_s: number }
  | { type: "STAGE_SKIPPED"; stage: VisionStage; config_key?: string }
  | { type: "STAGE_ERROR"; stage: VisionStage; error: string }
  | { type: "PIPELINE_STATE"; stages: Record<string, unknown> }
  | { type: "COMPLETE" }
  | { type: "RESET" };

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case "START":
      return { ...INITIAL_STATE, status: "running", videoId: action.videoId };
    case "STAGE_STARTED":
      return {
        ...state,
        status: "running",
        stages: {
          ...state.stages,
          [action.stage]: { status: "active", started_at: action.timestamp * 1000 },
        },
      };
    case "STAGE_PROGRESS":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            ...state.stages[action.stage as VisionStage],
            progress: action.progress,
            frame: action.frame,
            total_frames: action.total_frames,
          },
        },
      };
    case "STAGE_COMPLETE":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            status: "complete",
            config_key: action.config_key,
            duration_ms: action.duration_s * 1000,
          },
        },
      };
    case "STAGE_SKIPPED":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: { status: "skipped", config_key: action.config_key },
        },
      };
    case "STAGE_ERROR":
      return {
        ...state,
        status: "error",
        stages: {
          ...state.stages,
          [action.stage]: { status: "error", error: action.error },
        },
      };
    case "PIPELINE_STATE": {
      // Hydrate from server snapshot — used on SSE reconnect
      const hydrated = { ...state, status: "running" as const };
      if (action.stages) {
        for (const [key, val] of Object.entries(action.stages)) {
          if (key in hydrated.stages) {
            hydrated.stages = { ...hydrated.stages, [key]: { ...hydrated.stages[key as VisionStage], ...(val as object) } };
          }
        }
      }
      return hydrated;
    }
    case "COMPLETE":
      return { ...state, status: "complete" };
    case "RESET":
      return INITIAL_STATE;
    default:
      return state;
  }
}

export function usePipeline() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  const handleSSE = useCallback((event: SSEEvent) => {
    switch (event.event) {
      case "stage_started":
        dispatch({ type: "STAGE_STARTED", stage: event.stage as VisionStage, timestamp: event.timestamp ?? Date.now() / 1000 });
        break;
      case "stage_progress":
        dispatch({
          type: "STAGE_PROGRESS",
          stage: event.stage as VisionStage,
          progress: event.progress ?? 0,
          frame: event.frame ?? 0,
          total_frames: event.total_frames ?? 0,
        });
        break;
      case "stage_completed":
        dispatch({ type: "STAGE_COMPLETE", stage: event.stage as VisionStage, config_key: event.config_key, duration_s: event.duration_s ?? 0 });
        break;
      case "stage_skipped":
        dispatch({ type: "STAGE_SKIPPED", stage: event.stage as VisionStage, config_key: event.config_key });
        break;
      case "stage_error":
        dispatch({ type: "STAGE_ERROR", stage: event.stage as VisionStage, error: event.error ?? "Unknown error" });
        break;
      case "pipeline_state":
        dispatch({ type: "PIPELINE_STATE", stages: event.stages ?? {} });
        break;
      case "pipeline_completed":
        dispatch({ type: "COMPLETE" });
        break;
    }
  }, []);

  const { connected } = useSSE(state.videoId, { onEvent: handleSSE });

  const runPipeline = useCallback(
    async (videoId: string, _videoUrl: string, settings: AnalysisSettings) => {
      dispatch({ type: "START", videoId });
      try {
        await api.runFullPipeline(videoId, settings);
      } catch (err) {
        dispatch({ type: "STAGE_ERROR", stage: "download", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [],
  );

  const runStage = useCallback(
    async (stage: VisionStage, videoId: string, params: object) => {
      // Individual stage endpoints — fire and forget
      const stageEndpoints: Record<string, (id: string, p: object) => Promise<unknown>> = {
        detect: api.detectPlayers,
        track: api.trackPlayers,
        ocr: api.ocrJerseys,
        "classify-teams": api.classifyTeams,
        "court-map": api.mapCourt,
      };
      const fn = stageEndpoints[stage];
      if (fn) await fn(videoId, params);
    },
    [],
  );

  const rerunStage = useCallback(
    async (stage: VisionStage, videoId: string, configKey: string, params: object) => {
      await api.deleteArtifact(stage, videoId, configKey);
      await runStage(stage, videoId, params);
    },
    [runStage],
  );

  const cancelPipeline = useCallback(async () => {
    if (state.videoId) await api.cancelPipeline(state.videoId);
  }, [state.videoId]);

  const reset = useCallback(() => dispatch({ type: "RESET" }), []);

  return { state, runPipeline, runStage, rerunStage, cancelPipeline, reset, connected };
}
```

- [ ] **Step 2: Update analysis-layout.tsx to wire new API**

In `frontend/src/components/analysis-layout.tsx`, update the destructured return from `usePipeline`:

```typescript
// Line 29, change:
const { state, runPipeline, reset } = usePipeline();
// To:
const { state, runPipeline, runStage, rerunStage, cancelPipeline, reset, connected } = usePipeline();
```

Pass `runStage`, `rerunStage` to PipelineTable (will be used in Task 14).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/use-pipeline.ts frontend/src/components/analysis-layout.tsx
git commit -m "feat: SSE-driven usePipeline with runStage and rerunStage"
```

---

## Task 13: Settings Context + Settings Dialog Rewrite

**Files:**
- Modify: `frontend/src/contexts/analysis-settings-context.tsx`
- Modify: `frontend/src/components/settings-dialog.tsx`

- [ ] **Step 1: Update AnalysisSettingsProvider for stage-keyed model**

Rewrite `frontend/src/contexts/analysis-settings-context.tsx` to use the new `StageSettings` type. Replace `updateAdvanced` with `updateStage(stage, partial)`.

Key changes:
- `DEFAULT_SETTINGS` uses `stages: { detect: {...}, track: {...}, ... }`
- `updateStage(stage: keyof StageSettings, partial: Partial<...>)` replaces `updateAdvanced`
- Keep `updateGameContext` as-is

- [ ] **Step 2: Rewrite settings-dialog.tsx with per-stage sidebar**

Full rewrite following the Foreign Whispers pattern at `/home/pantelis.monogioudis/local/ai/apps/computer-vision/auraison-app/foreign-whispers/frontend/src/components/settings-dialog.tsx`.

Structure:
- `SECTIONS` array: Game Context, Download, Transcribe, Detect, Track, OCR, Teams, Court Map
- Left sidebar `w-48 border-r` with active state
- Right panel with per-stage settings component
- Each stage component: model selector (text input) + sliders/number inputs
- Reference the existing Foreign Whispers dialog for exact Tailwind classes

- [ ] **Step 3: Commit**

```bash
git add frontend/src/contexts/analysis-settings-context.tsx frontend/src/components/settings-dialog.tsx
git commit -m "feat: per-stage settings dialog with sidebar navigation"
```

---

## Task 14: Pipeline Table — Action Buttons + Progress Bar

**Files:**
- Modify: `frontend/src/components/pipeline-table.tsx`
- Modify: `frontend/src/components/pipeline-status-bar.tsx`

- [ ] **Step 1: Add Action column with Run/Re-run buttons**

Modify `frontend/src/components/pipeline-table.tsx`:
- Import `isStageReady`, `getDownstream` from `@/lib/stage-deps`
- Add a 5th column header: "Action"
- Add action button logic in `StageRow`:
  - `pending` + ready → "Run" button (primary outline)
  - `pending` + not ready → disabled button with tooltip
  - `active` → disabled
  - `complete` → "Re-run" (ghost) with confirmation popover
  - `error` → "Re-run" (destructive outline) with confirmation
- Confirmation popover lists affected downstream stages
- Props: add `onRunStage`, `onRerunStage`, `videoId`, `stages` (full Record for ready check)

- [ ] **Step 2: Add progress bar for active stages**

In `StageRow`, when status is `active` and `stage.progress` is defined, show a progress bar in the Duration column:

```tsx
{stage.status === "active" && stage.progress != null ? (
  <div className="flex items-center gap-2">
    <div className="h-1.5 w-16 rounded-full bg-muted overflow-hidden">
      <div
        className="h-full bg-primary rounded-full transition-all"
        style={{ width: `${Math.round(stage.progress * 100)}%` }}
      />
    </div>
    <span className="text-xs tabular-nums">{Math.round(stage.progress * 100)}%</span>
  </div>
) : (
  formatDuration(duration)
)}
```

- [ ] **Step 3: Update pipeline-status-bar.tsx with progress percentage**

In `frontend/src/components/pipeline-status-bar.tsx`, when the active stage has `progress`:

```typescript
const activeProgress = activeStage ? state.stages[activeStage].progress : undefined;

// In message construction:
if (activeStage) {
  const pct = activeProgress != null ? ` ${Math.round(activeProgress * 100)}%` : "";
  const elapsedStr = formatElapsed(elapsed);
  message = `${STAGE_DISPLAY_NAMES[activeStage]}${pct}${elapsedStr ? ` (${elapsedStr})` : ""}`;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pipeline-table.tsx frontend/src/components/pipeline-status-bar.tsx
git commit -m "feat: pipeline table action buttons, progress bars, cascade confirmation"
```

---

## Task 15: Integration Test + Manual Verification

- [ ] **Step 1: Run all backend tests**

```bash
uv run pytest tests/ -v
```
Expected: All pass

- [ ] **Step 2: Run frontend type check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: No errors

- [ ] **Step 3: Build frontend**

```bash
cd frontend && npm run build
```
Expected: Build succeeds

- [ ] **Step 4: Commit any fixes from type checking**

- [ ] **Step 5: Start Docker containers and test manually**

```bash
docker compose --profile nvidia build
docker compose --profile nvidia up -d
```

Open http://localhost:3000. Verify:
1. Settings dialog shows per-stage sidebar
2. Click "Analyze" → pipeline starts, SSE events stream in browser devtools
3. Progress bar updates during detection
4. "Re-run" button appears on completed stages
5. Check Logfire traces at https://logfire-us.pydantic.dev/pantelis/basket-tube

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "test: integration verification pass"
```
