# BasketTube Vision Pipeline API Design

**Date:** 2026-03-21
**Status:** Approved (rev 2)

## Overview

BasketTube analyzes basketball game video through a multi-stage computer vision pipeline. The system is split into four containers: a CPU API orchestrator, a Whisper STT service (speaches), a single GPU inference service, and a notebook service — communicating via HTTP with shared filesystem for large artifacts.

## Architecture

```
┌─────────────────────────┐       ┌──────────────────────────────┐
│   CPU API (:8080)       │ HTTP  │ inference (:8090)            │
│   FastAPI               │──────▶│ FastAPI                      │
│                         │       │                              │
│ • Vision router         │       │ • RF-DETR player detection   │
│ • Orchestration/caching │       │ • Court keypoint detection   │
│ • Video download/serve  │       │ • Jersey number OCR          │
│ • Frontend API          │       │ • SAM2 tracking/segmentation │
│                         │       │ • TeamClassifier (SigLIP)    │
│                         │       └──────────────────────────────┘
│                         │
│                         │ HTTP  ┌──────────────────────────────┐
│                         │──────▶│ whisper (:8000)              │
│                         │       │ speaches                     │
│                         │       │ • Commentary STT             │
└────────┬────────────────┘       └──────────┬───────────────────┘
         │                                   │
         └────────── ./pipeline_data ────────┘
                   (shared volume)
```

A fourth container (`notebook`) reuses `Dockerfile.gpu` with a JupyterLab CMD for prototyping.

## Identity Model

**`video_id`** is the YouTube video ID (e.g., `LPDnemFoqVk`). It is the primary key used in all API endpoints and artifact lookup.

**`stem`** is the slugified title from `video_registry.yml` (e.g., `Warriors & Lakers Instant Classic - 2021 Play-In Tournament`). It is used only for human-readable filenames on disk.

**Resolution:** A single shared function `resolve_stem(video_id) → stem` (in `api/src/video_registry.py`) performs the lookup. All services — CPU API and GPU services — use this function. GPU services receive `video_id` over HTTP and resolve to filesystem paths internally. The registry YAML is mounted read-only into all containers.

**Artifact lookup:** Given a `video_id`, a stage name, and a config key, the canonical artifact path is:

```
{data_dir}/analysis/{stage}/{config_key}/{stem}.json
```

No service passes raw filesystem paths over HTTP. Instead, requests contain `video_id` and parameters; each service resolves paths locally.

## Artifact Isolation & Caching

Stages with overridable parameters or swappable models must isolate their outputs by configuration. A **config key** is a short deterministic hash of the parameters that affect the output. This follows the existing TTS pipeline pattern (`c-{hash}`).

### Config Key Computation

Each stage computes a config key from its request parameters:

```python
import hashlib, json

def config_key(params: dict) -> str:
    """Deterministic short hash of parameters that affect output."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "c-" + hashlib.sha256(canonical.encode()).hexdigest()[:7]
```

### Directory Layout

```
pipeline_data/api/
├── videos/{stem}.mp4                              # source video
├── analysis/
│   ├── detections/{config_key}/{stem}.json         # stage 1
│   ├── tracks/{config_key}/{stem}.json             # stage 2
│   ├── teams/{config_key}/{stem}.json              # stage 3
│   ├── jerseys/{config_key}/{stem}.json            # stage 4
│   ├── court/{config_key}/{stem}.json              # stage 5
│   └── renders/{config_key}/{stem}.mp4             # stage 6
```

**Example:** Detection with `confidence=0.4, iou_threshold=0.9, model=basketball-player-detection-3-ycjdo/4` hashes to `c-a3f82b1`. Changing confidence to `0.3` produces `c-7e19d04` — a separate directory, no stale cache hit.

### Skip-on-Exists

If the output file exists for the given config key, the endpoint returns immediately with `skipped: true`. Changing any parameter produces a new config key and triggers reprocessing. Downstream stages that depend on a specific upstream config key must record which upstream config they consumed (stored in the output JSON metadata).

### Cross-Stage Dependencies

Each stage's output JSON includes a `_meta` field recording the config keys of its inputs:

```json
{
  "_meta": {
    "stage": "tracks",
    "config_key": "c-b2a91e3",
    "upstream": {
      "detections": "c-a3f82b1"
    },
    "created_at": "2026-03-21T14:30:00Z"
  },
  "frames": [...]
}
```

This enables invalidation: if detection is re-run with different parameters, the track stage knows its upstream changed.

## Pipeline Stages

All paths below are relative to the shared `pipeline_data/` mount. The existing `data_dir` in `api/src/config.py` resolves to `pipeline_data/api/` — vision artifacts live under `pipeline_data/api/analysis/` to stay consistent with the existing directory structure.

### Stage 1: Detection

- **Endpoint (CPU API):** `POST /api/vision/detect/{video_id}`
- **Delegates to:** `inference POST /api/detect`
- **Model:** `basketball-player-detection-3-ycjdo/4` (RF-DETR)
- **Input:** `videos/{stem}.mp4`
- **Output:** `analysis/detections/{config_key}/{stem}.json`
- **Config key inputs:** `{model_id, confidence, iou_threshold}`
- **Content:** Per-frame bounding boxes, class IDs, confidence scores
- **Default parameters:** confidence=0.4, iou_threshold=0.9
- **Class IDs:** 0=ball, 1=ball-in-basket, 2=number, 3=player, 4=player-in-possession, 5=jump-shot, 6=layup-dunk, 7=shot-block

### Stage 2: Tracking

- **Endpoint (CPU API):** `POST /api/vision/track/{video_id}`
- **Delegates to:** `inference POST /api/track`
- **Model:** SAM2.1 Large (`sam2.1_hiera_large.pt`)
- **Input:** `videos/{stem}.mp4` + `analysis/detections/{det_config_key}/{stem}.json`
- **Output:** `analysis/tracks/{config_key}/{stem}.json`
- **Config key inputs:** `{sam2_checkpoint, det_config_key}`
- **Content:** Per-frame masks (RLE-encoded), tracker IDs, bounding boxes
- **Dependencies:** Stage 1

### Stage 3: Team Classification

- **Endpoint (CPU API):** `POST /api/vision/classify-teams/{video_id}`
- **Delegates to:** `inference POST /api/classify-teams`
- **Model:** SigLIP embeddings + UMAP + K-means (via `sports.TeamClassifier`)
- **Input:** `videos/{stem}.mp4` + `analysis/detections/{det_config_key}/{stem}.json`
- **Output:** `analysis/teams/{config_key}/{stem}.json`
- **Config key inputs:** `{stride, crop_scale, det_config_key}`
- **Default parameters:** stride=30, crop_scale=0.4, k=2
- **Dependencies:** Stage 1

**Output JSON schema:**

```json
{
  "_meta": { "stage": "teams", "config_key": "...", "upstream": {"detections": "..."} },
  "palette": {
    "0": {"name": "Team A", "color": "#006BB6"},
    "1": {"name": "Team B", "color": "#007A33"}
  },
  "assignments": [
    {"frame_index": 0, "detection_index": 3, "team_id": 0},
    {"frame_index": 0, "detection_index": 5, "team_id": 1}
  ]
}
```

The classifier operates on per-crop embeddings sampled at `stride` intervals. It does not require tracker IDs. The `assignments` array maps `(frame_index, detection_index)` → `team_id`. At render time, if tracks are available, team labels are propagated to tracker IDs by joining on `(frame_index, detection_index)`.

The `palette` section holds team names and colors. These are initially auto-assigned (team 0/1) and can be manually corrected via a future endpoint or by editing the JSON.

### Stage 4: Jersey Number OCR

- **Endpoint (CPU API):** `POST /api/vision/ocr/{video_id}`
- **Delegates to:** `inference POST /api/ocr`
- **Model:** `basketball-jersey-numbers-ocr/3` (SmolVLM2)
- **Input:** `videos/{stem}.mp4` + `analysis/tracks/{track_config_key}/{stem}.json`
- **Output:** `analysis/jerseys/{config_key}/{stem}.json`
- **Config key inputs:** `{model_id, n_consecutive, ocr_interval, track_config_key}`
- **Content:** Validated jersey numbers per tracker ID (consecutive agreement n=3)
- **Dependencies:** Stage 2 (needs tracker IDs for temporal validation)

### Stage 5: Court Mapping

- **Endpoint (CPU API):** `POST /api/vision/court-map/{video_id}`
- **Delegates to:** `inference POST /api/keypoints`
- **CPU computes:** Homography from keypoints + player position transform
- **Model:** `basketball-court-detection-2/14` (keypoint detection)
- **Input:** `videos/{stem}.mp4` + `analysis/detections/{det_config_key}/{stem}.json`
- **Output:** `analysis/court/{config_key}/{stem}.json`
- **Config key inputs:** `{model_id, keypoint_confidence, anchor_confidence, det_config_key}`
- **Content:** Per-frame court XY positions (feet), cleaned trajectories
- **Default parameters:** keypoint_confidence=0.3, anchor_confidence=0.5
- **Dependencies:** Stage 1

### Stage 5b: Ball Tracking & Possession (future)

> **Not in the initial implementation plan.** Defined here to lock in the contract for the action recognition dataset spec.

- **Endpoint (CPU API):** `POST /api/vision/ball-track/{video_id}`
- **Delegates to:** `inference POST /api/ball-track` (or derived CPU-side from detection output)
- **Input:** `analysis/detections/{det_config_key}/{stem}.json` + `analysis/tracks/{track_config_key}/{stem}.json`
- **Output:** `analysis/ball_tracks/{config_key}/{stem}.json`
- **Config key inputs:** `{det_config_key, track_config_key}`
- **Content:**
  - Per-frame ball position (from class_id 0 detections)
  - Ball height/arc estimation
  - Player-to-ball proximity per frame
  - Possession state (which tracker_id is closest/possessing)
  - Possession change events with timestamps
- **Dependencies:** Stages 1, 2

This stage is required by the action recognition dataset builder for grounding shot attempts, rebounds, steals, and possession-based labels.

### Stage 6: Render

- **Endpoint (CPU API):** `POST /api/vision/render/{video_id}`
- **Runs on:** CPU (ffmpeg/opencv, no GPU)
- **Input:** All previous stage JSONs (resolved by their config keys) + source video
- **Output:** `analysis/renders/{config_key}/{stem}.mp4`
- **Config key inputs:** `{det_config_key, track_config_key, teams_config_key, jerseys_config_key, court_config_key}`
- **Content:** Annotated video with team-colored masks, jersey labels, court overlay
- **Dependencies:** Stages 1-5

### Stage Dependencies

```
detect (1)
  ├── track (2) ──────────┐
  │     └── ocr (4) ──────┤
  ├── classify-teams (3) ──┼──▶ render (6)
  └── court-map (5) ──────┘
```

Stage 4 (OCR) depends on stage 2 (tracking) because jersey number validation uses tracker IDs for temporal consistency. Stage 6 (render) depends on all previous stages.

If a prerequisite is missing, the CPU API returns HTTP 409 with a JSON body indicating which stage(s) must be run first: `{"detail": "Stage 'track' must be completed before 'ocr'", "missing": ["track"]}`.

## Execution Model

The CPU API calls stages **sequentially by default**. It does not issue parallel requests to the GPU service. This is intentional:

- All GPU stages hit the single `inference` container (:8090). Running them concurrently would require the GPU container to handle concurrent CUDA workloads, with GPU memory budgeting and request queuing — unnecessary complexity for a single-host deployment.
- The natural call order is: detect → track → classify-teams → ocr → court-map → render. Each stage is fast to skip if cached.

Parallel execution is a future optimization. It would require adding a request queue (e.g., FastAPI BackgroundTasks or Celery) to the GPU container, with `CUDA_MEM_FRACTION` limits to prevent OOM.

## Status & Lifecycle

### Status Model

Each stage maintains a status sidecar file alongside its output:

```
analysis/detections/{config_key}/{stem}.status.json
```

Status values align with the frontend's existing `StageStatus` type (`frontend/src/lib/types.ts:53`):

| Status | Meaning |
|---|---|
| `pending` | No output or status file exists |
| `active` | Processing started, not yet complete |
| `complete` | Output file exists and is valid |
| `skipped` | Output file existed before this request (cache hit) |
| `error` | Processing failed; error message recorded |

### Status Sidecar Format

```json
{
  "status": "complete",
  "started_at": "2026-03-21T14:30:00Z",
  "completed_at": "2026-03-21T14:31:42Z",
  "duration_ms": 102000,
  "error": null,
  "config_key": "c-a3f82b1"
}
```

Written atomically (write `.tmp`, rename) at:
- **Start:** `{"status": "active", "started_at": "...", ...}`
- **Completion:** `{"status": "complete", "completed_at": "...", "duration_ms": ..., ...}`
- **Failure:** `{"status": "error", "error": "...", ...}`

### Status Endpoint

- **Endpoint:** `GET /api/vision/status/{video_id}`
- **Query:** `config_key: str | None` — if provided, checks status for that specific config. If omitted, returns the latest (most recent `completed_at`) config per stage.
- **Response:**

```python
class StageStatusResponse(BaseModel):
    status: str          # "pending" | "active" | "complete" | "skipped" | "error"
    config_key: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error: str | None = None

class PipelineStatusResponse(BaseModel):
    video_id: str
    stages: dict[str, StageStatusResponse]
```

### Crash Recovery

If a stage crashes (container restart, OOM kill), the status sidecar remains `"active"` with a `started_at` but no `completed_at`. The CPU API detects this as stale if `started_at` is older than the GPU service timeout (600s) and treats it as `"error"` with a synthetic message. The stale `.status.json` is removed so the stage can be retried.

## Error Handling

- **GPU service unreachable:** CPU API returns HTTP 502 with the connection error. Timeout for GPU service calls: 600 seconds (video processing is slow).
- **Atomic writes:** GPU services write results to a temporary file (`.tmp` suffix) and rename on completion. This prevents corrupt cache hits if a job fails mid-write. Status sidecars follow the same pattern.
- **Partial failure:** If a GPU service returns an error, the CPU API writes an `error` status sidecar and forwards the error as HTTP 500 with the GPU service's error message. No output file is written, so retry will reprocess.
- **409 responses:** Include which prerequisite stage(s) are missing (see above).
- **Concurrent requests for same video_id + config_key:** The status sidecar's `active` state acts as a lightweight lock. If a request arrives while a stage is already `active`, the CPU API returns HTTP 409 with `"detail": "Stage 'detect' is already running"`.

## Data Exchange

CPU API and GPU services communicate via:
- **HTTP** for triggering work (lightweight JSON payloads: `video_id`, parameters — no raw filesystem paths)
- **Shared filesystem** (`./pipeline_data`) for large artifacts (video files, result JSONs)

No frames or crops are sent over HTTP. Each service resolves `video_id` → filesystem paths internally using the shared `video_registry.yml` and its own `data_dir` config.

### GPU Service Request/Response (internal)

```python
class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}              # stage-specific parameters
    upstream_configs: dict = {}    # e.g., {"detections": "c-a3f82b1"}

class InferenceResponse(BaseModel):
    status: str                    # "ok" or "error"
    config_key: str                # computed config key for this run
    output_path: str               # relative to data_dir
    error: str | None = None
```

## Schemas

### CPU API (api/src/schemas/vision.py)

```python
class DetectRequest(BaseModel):
    model_id: str = "basketball-player-detection-3-ycjdo/4"
    confidence: float = 0.4
    iou_threshold: float = 0.9
    max_frames: int | None = None

class TrackRequest(BaseModel):
    det_config_key: str            # which detection run to use
    max_frames: int | None = None

class ClassifyTeamsRequest(BaseModel):
    det_config_key: str
    stride: int = 30
    crop_scale: float = 0.4

class OCRRequest(BaseModel):
    track_config_key: str          # which tracking run to use
    model_id: str = "basketball-jersey-numbers-ocr/3"
    n_consecutive: int = 3
    ocr_interval: int = 5

class CourtMapRequest(BaseModel):
    det_config_key: str
    model_id: str = "basketball-court-detection-2/14"
    keypoint_confidence: float = 0.3
    anchor_confidence: float = 0.5

class RenderRequest(BaseModel):
    det_config_key: str
    track_config_key: str
    teams_config_key: str
    jerseys_config_key: str
    court_config_key: str

class StageStatusResponse(BaseModel):
    status: str
    config_key: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error: str | None = None

class DetectResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames: int
    n_detections: int
    skipped: bool = False

class TrackResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames: int
    n_tracks: int
    skipped: bool = False

class ClassifyTeamsResponse(BaseModel):
    video_id: str
    config_key: str
    palette: dict[str, dict]       # {"0": {"name": "...", "color": "#..."}}
    skipped: bool = False

class OCRResponse(BaseModel):
    video_id: str
    config_key: str
    players: dict[str, str]        # {"tracker_id": "jersey_number"}
    skipped: bool = False

class CourtMapResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames_mapped: int
    skipped: bool = False

class RenderResponse(BaseModel):
    video_id: str
    config_key: str
    skipped: bool = False

class PipelineStatusResponse(BaseModel):
    video_id: str
    stages: dict[str, StageStatusResponse]
```

## Configuration

### CPU API (FW_ prefix)

The CPU API retains the `FW_` env prefix from the inherited foreign-whispers codebase. This prefix is kept for backward compatibility with the existing dubbing pipeline settings. New vision-specific settings are added under the same prefix.

New settings in `api/src/config.py`:

| Setting | Env Var | Default |
|---|---|---|
| `inference_gpu_url` | `FW_INFERENCE_GPU_URL` | `http://localhost:8090` |
| `whisper_api_url` | `FW_WHISPER_API_URL` | `http://localhost:8000` |
| `analysis_dir` | computed | `data_dir / "analysis"` |

### GPU inference (BT_ prefix)

The GPU inference service uses the `BT_` (BasketTube) prefix to distinguish from the CPU API's `FW_` settings.

| Env Var | Purpose | Default |
|---|---|---|
| `ROBOFLOW_API_KEY` | API key | required |
| `INFERENCE_MODE` | `local` or `remote` | `local` |
| `SAM2_REPO` | SAM2 install path | `/opt/segment-anything-2-real-time` |
| `HF_TOKEN` | SigLIP model downloads | required |
| `BT_DATA_DIR` | Pipeline data root | `/app/pipeline_data` |

## Docker

### Dockerfiles

Two Dockerfiles are used:

| File | Base Image | Purpose |
|---|---|---|
| `Dockerfile.api` | `python:3.11-slim` | CPU API — lightweight orchestrator with FastAPI, no GPU deps |
| `Dockerfile.gpu` | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` | Single GPU inference service — `inference-gpu`, `supervision`, `sports`, SAM2, `jupyterlab` |

The `notebook` service reuses `Dockerfile.gpu` with a different CMD (`jupyter lab`). The notebook dependency group from `pyproject.toml` (`jupyterlab`, `ipywidgets`) is installed in `Dockerfile.gpu`.

All containers mount `video_registry.yml` read-only for `resolve_stem()`.

### Docker Compose

All services use `network_mode: host` for simplicity on single-host deployments. With host networking, port mappings are implicit (each service binds directly to its port).

```yaml
services:
  api:
    build: { dockerfile: Dockerfile.api }
    profiles: [nvidia, cpu]
    network_mode: host
    # Binds to :8080
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro

  whisper:
    image: ghcr.io/speaches-ai/speaches:latest
    profiles: [nvidia]
    network_mode: host
    # Binds to :8000

  inference:
    build: { dockerfile: Dockerfile.gpu }
    profiles: [nvidia]
    network_mode: host
    shm_size: "8gb"
    # GPU reservation
    # Binds to :8090
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro

  notebook:
    build: { dockerfile: Dockerfile.gpu }
    profiles: [nvidia]
    network_mode: host
    shm_size: "8gb"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", ...]
    # GPU reservation
    # Binds to :8888
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro
      - ./notebooks:/workspace/notebooks
```

## New Code

| File | Purpose |
|---|---|
| `basket_tube/inference/main.py` | FastAPI app with /api/detect, /api/keypoints, /api/ocr, /api/track, /api/classify-teams |
| `basket_tube/inference/roboflow/` | Roboflow model loading and inference wrappers |
| `basket_tube/inference/vision/tracker.py` | SAM2Tracker class (extracted from notebook) |
| `basket_tube/inference/vision/classifier.py` | TeamClassifier wrapper |
| `api/src/routers/vision.py` | CPU API vision pipeline router (6 stage endpoints + status) |
| `api/src/schemas/vision.py` | Per-stage request/response Pydantic models |
| `api/src/services/vision_service.py` | HTTP client to GPU inference service (httpx async) |
| `api/src/services/whisper_service.py` | Remote HTTP client for Whisper STT (speaches container) |
| `Dockerfile.api` | CPU API Dockerfile |
| `Dockerfile.gpu` | GPU Dockerfile for all inference + notebook |

## Existing Code Modified

| File | Change |
|---|---|
| `api/src/config.py` | Add `inference_gpu_url`, `whisper_api_url`, `analysis_dir` |
| `api/src/main.py` | Register download, transcribe, vision, and captions routers |
| `api/src/video_registry.py` | Add `resolve_stem()` alias (if not already present) |

## Existing Code Unchanged

The existing download and transcribe routers, services, and Whisper backend remain functional for the commentary analysis side of BasketTube.
