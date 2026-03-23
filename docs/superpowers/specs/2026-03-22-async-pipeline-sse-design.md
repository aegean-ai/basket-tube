# Async Pipeline with SSE Progress & Per-Stage Settings

**Date:** 2026-03-22
**Status:** Draft

## Problem

BasketTube's pipeline has four blocking issues:

1. **Event loop blocking** — `DownloadService` (yt_dlp) and `WhisperService` (requests.post) are synchronous, freezing the FastAPI server for minutes.
2. **No real-time feedback** — The frontend awaits each stage sequentially. No SSE or WebSocket. The UI freezes during long GPU operations.
3. **Missing instrumentation** — Download, text timeline, and pipeline orchestration are not traced via Logfire.
4. **Flat settings UI** — All pipeline parameters are in a single flat dialog. No per-stage organization, no model configuration.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Real-time transport | SSE (Server-Sent Events) | One-directional server→client fits; simpler than WebSocket |
| Pipeline trigger | Hybrid: full-pipeline endpoint + individual re-run endpoints | Single call for common case; per-stage for re-runs |
| Progress granularity | Stage + percentage (frame N/total) | GPU service writes progress files; API polls and forwards via SSE |
| Blocking download fix | `asyncio.to_thread()` for yt_dlp | Complex sync library, no async alternative |
| Blocking whisper fix | Replace `requests` with `httpx.AsyncClient` | Simple HTTP call, consistent with VisionService |
| Settings UI pattern | Per-stage sidebar dialog (Foreign Whispers pattern) | Each stage gets its own settings panel with model config |

## Architecture

### Stage Dependency Graph

```
download → transcribe (independent audio path, no GPU)

download → detect ─┬→ track → ocr
                    ├→ classify-teams
                    └→ court-map
                                    ──→ render (requires track + ocr + classify-teams + court-map)
```

**Key:** `classify-teams` and `court-map` depend on `detect` (not `track`), so they can start immediately after detect completes — concurrently with `track`. Only `ocr` must wait for `track`.

**Orchestrator scheduling:**
```python
# After detect completes:
track_task = asyncio.create_task(run_track(...))
teams_task = asyncio.create_task(run_classify_teams(...))
court_task = asyncio.create_task(run_court_map(...))

# OCR must wait for track:
track_result = await track_task
ocr_task = asyncio.create_task(run_ocr(...))

# Wait for all parallel branches:
await asyncio.gather(teams_task, court_task, ocr_task)
# Now render can proceed
```

**Note:** `transcribe` runs independently on the audio path. It is not part of the vision pipeline DAG and is not included in the SSE-driven pipeline orchestrator. It remains a separate `POST /api/transcribe/{video_id}` call.

### SSE Event Protocol

**Endpoint:** `GET /api/pipeline/events/{video_id}`

Returns `text/event-stream` with the following event types:

```
event: pipeline_state
data: {"stages": {"detect": {"status": "complete", ...}, "track": {"status": "active", ...}, ...}}
(Sent on initial SSE connection to hydrate client state — solves browser refresh recovery)

event: stage_started
data: {"stage": "detect", "timestamp": 1711100000}

event: stage_progress
data: {"stage": "detect", "progress": 0.45, "frame": 270, "total_frames": 600}

event: stage_completed
data: {"stage": "detect", "duration_s": 342.1}

event: stage_skipped
data: {"stage": "detect", "config_key": "c-baf1922"}

event: stage_error
data: {"stage": "ocr", "error": "GPU OOM", "timestamp": 1711100500}

event: pipeline_completed
data: {"duration_s": 1200.5, "stages_completed": 7, "stages_skipped": 1}

: keepalive
(Comment line sent every 15 seconds to prevent proxy/browser timeout during long stages)
```

### SSE Recovery on Browser Refresh

When the `EventSource` connects (or reconnects), the server immediately sends a `pipeline_state` snapshot event containing the current status of all stages. This ensures the frontend can hydrate its state without missing events. The existing `GET /api/vision/status/{video_id}` endpoint provides this data — the SSE endpoint reuses that logic internally on connection.

### Concurrent Pipeline Run Policy

Only one pipeline run per `video_id` is allowed at a time. If the user triggers a new run while one is active:
- `POST /api/pipeline/run/{video_id}` returns `409 Conflict` with `{"detail": "Pipeline already running for this video", "sse_url": "/api/pipeline/events/{video_id}"}`
- The frontend shows an inline message and can reconnect to the existing SSE stream

To restart, the user must either wait for completion or cancel the active run via:
```
POST /api/pipeline/cancel/{video_id}
Response: 200 { "cancelled_stages": ["track", "ocr"] }
```
This calls `asyncio.Task.cancel()` on the running task and marks active stages as `error` with reason "cancelled".

### GPU Progress Reporting

The GPU inference service writes a progress file during frame processing:

```json
// pipeline_data/api/analysis/detections/c-baf1922/_progress.json
{"frame": 270, "total_frames": 600, "updated_at": 1711100123.4}
```

**Atomic writes:** The GPU service writes to a temporary file (`_progress.tmp`) and renames it to `_progress.json` to prevent partial reads by the API orchestrator.

The API orchestrator runs a progress poller as a separate `asyncio.Task` alongside the GPU HTTP call:

```python
async def _run_stage_with_progress(self, stage, gpu_coro, progress_path, event_queue):
    poller = asyncio.create_task(self._poll_progress(stage, progress_path, event_queue))
    try:
        result = await gpu_coro
    finally:
        poller.cancel()
        progress_path.unlink(missing_ok=True)
    return result
```

The poller reads the progress file every 2 seconds and emits `stage_progress` SSE events.

### Pipeline Orchestrator

New service: `api/src/services/pipeline_orchestrator.py`

Responsibilities:
- Accept a pipeline run request and create an `asyncio.Task` for execution
- Schedule stages respecting the dependency graph (see scheduling code above)
- Compute `config_key` server-side from stage parameters using `artifacts.config_key()`
- Pass upstream config keys between stages (detect's key flows to track, classify-teams, court-map; track's key flows to ocr)
- Maintain an in-memory event bus per `video_id` (`asyncio.Queue` of SSE events)
- Run a progress poller task alongside each GPU call
- Handle errors: mark failed stage + all downstream as error
- Handle cancellation: cancel the asyncio.Task, mark active stages as error

State management:
- Active pipeline runs stored in a module-level `dict[str, PipelineRun]`
- Each `PipelineRun` holds the asyncio.Task, event queue, stage states, and computed config keys
- Cleanup on pipeline completion, error, or cancellation
- On server restart, stale "active" sidecars are cleaned up by the existing `check_stale()` mechanism (10-minute timeout)

### Full Pipeline Endpoint

```
POST /api/pipeline/run/{video_id}
Body: { "settings": AnalysisSettings }
Response: 202 Accepted { "sse_url": "/api/pipeline/events/{video_id}" }
```

Kicks off the full pipeline as a background task. Returns immediately.

Optional `from_stage` parameter to re-run from a specific stage (cascade):
```
POST /api/pipeline/run/{video_id}?from_stage=ocr
```
This deletes artifacts for `ocr` and all downstream stages, then re-runs from `ocr` **using existing upstream artifacts with their current config keys**. The orchestrator reads the upstream sidecar files to discover the config keys that were used in the previous run.

### Individual Stage Re-run

Existing endpoints (`POST /api/vision/detect/{video_id}`, etc.) are modified to:
1. Return `202 Accepted` with body `{ "stage": "detect", "config_key": "c-...", "sse_url": "/api/pipeline/events/{video_id}" }`
2. Run the GPU call as a background `asyncio.Task`
3. Emit SSE events to the same event bus

The response body no longer contains result data (e.g., `n_frames`, `n_detections`). Results are delivered via the `stage_completed` SSE event and can be fetched from `GET /api/vision/artifacts/{stage}/{video_id}`.

### Artifact Deletion Endpoint

```
DELETE /api/vision/artifacts/{stage}/{video_id}?config_key=...
```

Deletes the artifact JSON, sidecar status file, and progress file for the specified stage. Called before re-running a stage.

## Async Fixes

### WhisperService (sync → async)

**Before** (`api/src/services/whisper_service.py`):
```python
response = requests.post(url, files={"file": ...}, timeout=600)
```

**After:**
```python
async def transcribe(audio_path: str) -> dict:
    async with httpx.AsyncClient(timeout=600) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            response = await client.post(url, files=files, data=data)
    response.raise_for_status()
    return response.json()
```

The Whisper model name is now passed dynamically from `settings.stages.transcribe.model` instead of reading from `config.py`'s fixed `whisper_model` setting. The `config.py` `whisper_model` field becomes the default fallback.

### DownloadService (sync → thread-offloaded)

**Before** (sync calls to yt_dlp in endpoint handler):
```python
result = download_service.download_video(url)
```

**After:**
```python
result = await asyncio.to_thread(download_service.download_video, url)
```

The download service itself stays synchronous; only the call site wraps it in `to_thread()`.

### build_timeline (sync → thread-offloaded)

Same pattern: `await asyncio.to_thread(build_timeline, ...)` in the captions router.

## Logfire Instrumentation

### New Spans

| Span Name | Location | Metadata |
|-----------|----------|----------|
| `pipeline.run` | pipeline_orchestrator.py | video_id, settings hash, total stages |
| `pipeline.stage.{name}` | pipeline_orchestrator.py (child of pipeline.run) | stage name, config_key, params |
| `pipeline.download` | download router | video_id, source (youtube/local) |
| `pipeline.transcribe` | transcribe router | video_id, model, source (whisper/youtube) |
| `pipeline.timeline` | captions router | video_id, n_segments |
| `pipeline.progress_poll` | pipeline_orchestrator.py | stage, frame, total_frames |

### WhisperService Cleanup

Replace manual `ctx.__enter__()` / `ctx.__exit__()` pattern with proper async context manager:
```python
with logfire.span("whisper.transcribe", filename=filename, model=model):
    response = await client.post(...)
```

## Per-Stage Settings UI

### Layout

Replaces the current flat `SettingsDialog` with the Foreign Whispers two-column pattern:

- **Dialog:** `w-[720px] max-h-[80vh]`, inner height `h-[560px]`
- **Left sidebar:** `w-48 border-r`, lists pipeline stages with icons
- **Right panel:** `flex-1 overflow-y-auto p-6`, shows selected stage's settings

### Settings Sections

Each stage maps to a dedicated settings component:

#### Game Context (UsersRoundIcon) — top of sidebar
- Team names and colors (2 teams)
- Player roster (jersey number → name)
- Not tied to a specific pipeline stage — metadata used by multiple stages

#### Download (DownloadIcon)
- Source URL / local file path (read-only display)
- Video quality badge: "Best available"

#### Transcribe (MicIcon)
- **Model selector:** Whisper variant (tiny / base / small / medium / large-v3)
- **Toggle:** Use YouTube captions when available

#### Detect (ScanSearchIcon)
- **Model selector:** RF-DETR model ID (text input, default: `basketball-player-detection-3-ycjdo/4`)
- **Slider:** Confidence threshold (0.0–1.0, default: 0.40)
- **Slider:** IOU threshold (0.0–1.0, default: 0.90)

#### Track (RouteIcon)
- **Badge:** ByteTrack (algorithmic, no model)
- **Slider:** IOU threshold
- **Number input:** Track activation threshold
- **Number input:** Lost track buffer (frames)

#### OCR (HashIcon)
- **Model selector:** VLM model ID (text input, default: `basketball-jersey-numbers-ocr/3`)
- **Number input:** OCR interval (frames, default: 5)
- **Number input:** Consecutive reads threshold (default: 3)

#### Teams (UsersIcon)
- **Model selector:** Embedding model for crop features (text input, default: SigLIP variant)
- **Number input:** Number of teams (default: 2)
- **Slider:** Crop scale (0.0–1.0, default: 0.40)
- **Number input:** Sampling stride (frames, default: 30)

#### Court Map (MapIcon)
- **Model selector:** Keypoint model ID (text input, default: `basketball-court-detection-2/14`)
- **Slider:** Keypoint confidence (0.0–1.0, default: 0.30)
- **Slider:** Anchor confidence (0.0–1.0, default: 0.50)

### Settings Data Model

The flat `AdvancedSettings` is replaced with a stage-keyed structure:

```typescript
export interface StageSettings {
  download: {
    // Read-only, no configurable params
  };
  transcribe: {
    model: string;
    use_youtube_captions: boolean;
  };
  detect: {
    model_id: string;
    confidence: number;
    iou_threshold: number;
  };
  track: {
    iou_threshold: number;
    track_activation_threshold: number;
    lost_track_buffer: number;
  };
  ocr: {
    model_id: string;
    ocr_interval: number;
    n_consecutive: number;
  };
  teams: {
    embedding_model: string;
    n_teams: number;
    crop_scale: number;
    stride: number;
  };
  court_map: {
    model_id: string;
    keypoint_confidence: number;
    anchor_confidence: number;
  };
}

export interface AnalysisSettings {
  game_context: GameContext;
  stages: StageSettings;
}
```

The backend `PUT /api/settings/{video_id}` schema is updated to match. Existing `advanced` field is deprecated and migrated on load.

## Pipeline Table UI Changes

### Per-Stage Action Button

Each row in `PipelineTable` gets an action button column:

| Stage State | Button | Behavior |
|-------------|--------|----------|
| `pending` | Disabled, tooltip: "Waiting for {upstream}" | — |
| `ready` | "Run" (primary outline) | POST individual stage endpoint |
| `active` | Disabled | Shows progress bar instead |
| `complete` | "Re-run" (ghost) | Inline confirmation popover → DELETE artifacts → POST stage |
| `skipped` | "Run" (outline) | POST individual stage endpoint |
| `error` | "Re-run" (destructive outline) | Inline confirmation → DELETE → POST |

### New State: `ready`

A stage is `ready` when all upstream dependencies have status `complete` (or `skipped`) but the stage itself has not run. Computed client-side from the dependency graph.

### Progress Bar

When a stage is `active`, the Duration column shows a progress bar with percentage instead of elapsed time. Driven by `stage_progress` SSE events.

### Cascade Confirmation

Re-running a stage that has downstream dependents shows a popover:
> "This will also re-run: track, ocr, classify-teams, court-map. Continue?"

### Dependency Graph (client-side)

```typescript
const STAGE_DEPS: Record<VisionStage, VisionStage[]> = {
  "download": [],
  "transcribe": ["download"],
  "detect": ["download"],
  "track": ["detect"],
  "ocr": ["track"],
  "classify-teams": ["detect"],
  "court-map": ["detect"],
  "render": ["track", "ocr", "classify-teams", "court-map"],
};
```

Used to compute `ready` state and cascade invalidation lists.

**Note:** `download → detect` is an implicit dependency — detect needs the video file to exist, enforced by `_resolve_or_404()` rather than an artifact check.

## Frontend SSE Integration

### New Hook: `useSSE(videoId)`

```typescript
function useSSE(videoId: string | undefined) {
  // Creates EventSource to /api/pipeline/events/{videoId}
  // On connect: receives pipeline_state snapshot to hydrate initial state
  // Dispatches actions to pipeline reducer on each event
  // Auto-reconnects on connection loss (EventSource default behavior)
  // Returns: { connected: boolean, lastEvent: SSEEvent | null }
}
```

### Refactored `usePipeline`

- `runPipeline()` sends `POST /api/pipeline/run/{video_id}` and returns immediately
- `runStage(stage)` sends POST to individual endpoint and returns immediately
- `rerunStage(stage)` sends DELETE for artifacts, then POST
- `cancelPipeline()` sends POST to cancel endpoint
- State updates come exclusively from SSE events, not from awaiting API responses
- The reducer gains new actions: `STAGE_PROGRESS` (with progress/frame/total_frames), `PIPELINE_STATE` (hydrate from snapshot)

### API Module Changes

New functions in `api.ts`:
```typescript
export const runFullPipeline = (videoId: string, settings: AnalysisSettings) =>
  post<{ sse_url: string }>(`/api/pipeline/run/${videoId}`, { settings });

export const cancelPipeline = (videoId: string) =>
  post<{ cancelled_stages: string[] }>(`/api/pipeline/cancel/${videoId}`);

export const deleteArtifact = (stage: string, videoId: string, configKey: string) =>
  del(`/api/vision/artifacts/${stage}/${videoId}?config_key=${configKey}`);
```

## Files Changed

### New Files

| File | Purpose |
|------|---------|
| `api/src/routers/pipeline.py` | SSE endpoint, full-pipeline trigger, cancel endpoint |
| `api/src/services/pipeline_orchestrator.py` | Stage scheduling, progress polling, event bus |
| `api/src/schemas/pipeline.py` | Pipeline run request/response schemas, SSE event models |
| `frontend/src/hooks/use-sse.ts` | EventSource hook for SSE consumption |

### Modified Files — Backend

| File | Change |
|------|--------|
| `api/src/main.py` | Register pipeline router, add Logfire spans for download/timeline |
| `api/src/routers/vision.py` | Individual endpoints return 202 + run in background; add DELETE endpoint |
| `api/src/routers/download.py` | Wrap yt_dlp calls in `asyncio.to_thread()`; add Logfire span |
| `api/src/routers/transcribe.py` | Call async `whisper_service.transcribe()`; pass model from settings; add Logfire span |
| `api/src/routers/captions.py` | Wrap `build_timeline()` in `asyncio.to_thread()`; add Logfire span |
| `api/src/services/whisper_service.py` | Replace `requests` with `httpx.AsyncClient`; accept model param; clean up Logfire spans |
| `api/src/services/vision_service.py` | No callback needed — progress polling is handled by orchestrator |
| `api/src/schemas/vision.py` | Add `StageSettings` models per stage; update 202 response schemas |
| `api/src/artifacts.py` | Add `delete_artifact()` function for the DELETE endpoint |
| `api/src/config.py` | `whisper_model` becomes default fallback; dynamic model passed from settings |

### Modified Files — GPU Service

| File | Change |
|------|--------|
| `basket_tube/inference/main.py` | Write `_progress.json` (via atomic tmp+rename) during frame processing loops |

### Modified Files — Frontend

| File | Change |
|------|--------|
| `frontend/src/hooks/use-pipeline.ts` | SSE-driven state, `runStage()`, `rerunStage()`, `cancelPipeline()`, `STAGE_PROGRESS` + `PIPELINE_STATE` actions |
| `frontend/src/lib/api.ts` | Add `runFullPipeline()`, `cancelPipeline()`, `deleteArtifact()`, `del()` helper |
| `frontend/src/lib/types.ts` | Add `StageSettings`, update `AnalysisSettings`, add `ready` to `StageStatus`, add `progress`/`frame`/`total_frames` to `StageState` |
| `frontend/src/components/pipeline-table.tsx` | Add action button column, progress bar, cascade confirmation popover |
| `frontend/src/components/pipeline-status-bar.tsx` | Show % progress during active stages |
| `frontend/src/components/settings-dialog.tsx` | Complete rewrite: two-column layout with per-stage settings panels |
| `frontend/src/contexts/analysis-settings-context.tsx` | Update to stage-keyed settings model, migration from flat `advanced` |

## Migration

The backend settings endpoint detects the old flat `advanced` format and migrates it to the new `stages` structure on read. The old format is still accepted on write for backwards compatibility during the transition.

## Testing Strategy

1. **Unit tests:** Pipeline orchestrator stage scheduling, dependency graph, cascade invalidation
2. **Integration tests:** SSE event stream (FastAPI TestClient with `httpx` streaming), artifact deletion, concurrent run rejection
3. **Manual tests:** Run full pipeline, verify SSE events in browser devtools; re-run individual stages; browser refresh during pipeline; check Logfire traces
