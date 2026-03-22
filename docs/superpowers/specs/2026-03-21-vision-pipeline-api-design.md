# BasketTube Vision Pipeline API Design

**Date:** 2026-03-21
**Status:** Approved

## Overview

BasketTube analyzes basketball game video through a multi-stage computer vision pipeline. The system is split into three containers: a CPU API orchestrator and two GPU inference services, communicating via HTTP with shared filesystem for large artifacts.

## Architecture

```
┌─────────────────────────┐       ┌──────────────────────────────┐
│   CPU API (:8080)       │ HTTP  │ inference-roboflow (:8091)   │
│   FastAPI               │──────▶│ FastAPI                      │
│                         │       │                              │
│ • Vision router         │       │ • RF-DETR player detection   │
│ • Orchestration/caching │       │ • Court keypoint detection   │
│ • Video download/serve  │       │ • Jersey number OCR          │
│ • Commentary STT        │       │ • Mode: local GPU or remote  │
│ • Frontend API          │       │   Roboflow cloud API         │
│                         │       └──────────────────────────────┘
│                         │
│                         │       ┌──────────────────────────────┐
│                         │ HTTP  │ inference-vision (:8092)     │
│                         │──────▶│ FastAPI                      │
│                         │       │                              │
│                         │       │ • SAM2 tracking/segmentation │
│                         │       │ • TeamClassifier (SigLIP)    │
│                         │       │ • Always local GPU           │
└────────┬────────────────┘       └──────────┬───────────────────┘
         │                                   │
         └────────── ./pipeline_data ────────┘
                   (shared volume)
```

A fourth container (`notebook`) reuses `Dockerfile.vision` with a JupyterLab CMD for prototyping.

## Pipeline Stages

All paths below are relative to the shared `pipeline_data/` mount. The existing `data_dir` in `api/src/core/config.py` resolves to `pipeline_data/api/` — vision artifacts live under `pipeline_data/api/analysis/` to stay consistent with the existing directory structure.

### Stage 1: Detection

- **Endpoint (CPU API):** `POST /api/vision/detect/{video_id}`
- **Delegates to:** `inference-roboflow POST /api/detect`
- **Model:** `basketball-player-detection-3-ycjdo/4` (RF-DETR)
- **Input:** `pipeline_data/api/videos/{stem}.mp4`
- **Output:** `pipeline_data/api/analysis/detections/{stem}.json`
- **Content:** Per-frame bounding boxes, class IDs, confidence scores
- **Default parameters:** confidence=0.4, iou_threshold=0.9 (hardcoded defaults, overridable via request)
- **Class IDs:** 0=ball, 1=ball-in-basket, 2=number, 3=player, 4=player-in-possession, 5=jump-shot, 6=layup-dunk, 7=shot-block

### Stage 2: Tracking

- **Endpoint (CPU API):** `POST /api/vision/track/{video_id}`
- **Delegates to:** `inference-vision POST /api/track`
- **Model:** SAM2.1 Large (`sam2.1_hiera_large.pt`)
- **Input:** `pipeline_data/api/videos/{stem}.mp4` + `pipeline_data/api/analysis/detections/{stem}.json`
- **Output:** `pipeline_data/api/analysis/tracks/{stem}.json`
- **Content:** Per-frame masks (RLE-encoded), tracker IDs, bounding boxes
- **Dependencies:** Stage 1

### Stage 3: Team Classification

- **Endpoint (CPU API):** `POST /api/vision/classify-teams/{video_id}`
- **Delegates to:** `inference-vision POST /api/classify-teams`
- **Model:** SigLIP embeddings + UMAP + K-means (via `sports.TeamClassifier`)
- **Input:** `pipeline_data/api/videos/{stem}.mp4` + `pipeline_data/api/analysis/detections/{stem}.json`
- **Output:** `pipeline_data/api/analysis/teams/{stem}.json`
- **Content:** Team ID per tracker ID, team names/colors
- **Default parameters:** stride=30, crop_scale=0.4, k=2
- **Dependencies:** Stage 1
- **Note:** TeamClassifier operates on per-crop embeddings aggregated over sampled frames. It does not require tracker IDs — it clusters crops by visual appearance, then the result is keyed by detection index. If tracks are available, team labels can be propagated to tracker IDs at render time.

### Stage 4: Jersey Number OCR

- **Endpoint (CPU API):** `POST /api/vision/ocr/{video_id}`
- **Delegates to:** `inference-roboflow POST /api/ocr`
- **Model:** `basketball-jersey-numbers-ocr/3` (SmolVLM2)
- **Input:** `pipeline_data/api/videos/{stem}.mp4` + `pipeline_data/api/analysis/detections/{stem}.json`
- **Output:** `pipeline_data/api/analysis/jerseys/{stem}.json`
- **Content:** Validated jersey numbers per tracker ID (consecutive agreement n=3)
- **Dependencies:** Stage 2 (needs tracker IDs for temporal validation)

### Stage 5: Court Mapping

- **Endpoint (CPU API):** `POST /api/vision/court-map/{video_id}`
- **Delegates to:** `inference-roboflow POST /api/keypoints`
- **CPU computes:** Homography from keypoints + player position transform
- **Model:** `basketball-court-detection-2/14` (keypoint detection)
- **Input:** `pipeline_data/api/videos/{stem}.mp4` + `pipeline_data/api/analysis/detections/{stem}.json`
- **Output:** `pipeline_data/api/analysis/court/{stem}.json`
- **Content:** Per-frame court XY positions (feet), cleaned trajectories
- **Default parameters:** keypoint_confidence=0.3, anchor_confidence=0.5
- **Dependencies:** Stage 1

### Stage 6: Render

- **Endpoint (CPU API):** `POST /api/vision/render/{video_id}`
- **Runs on:** CPU (ffmpeg/opencv, no GPU)
- **Input:** All previous stage JSONs + source video
- **Output:** `pipeline_data/api/analysis/renders/{stem}.mp4`
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

Stages 2, 3, and 5 can run in parallel after stage 1 completes. Stage 4 (OCR) depends on stage 2 (tracking) because jersey number validation uses tracker IDs for temporal consistency. Stage 6 (render) depends on all previous stages.

If a prerequisite is missing, the CPU API returns HTTP 409 with a JSON body indicating which stage(s) must be run first: `{"detail": "Stage 'track' must be completed before 'ocr'", "missing": ["track"]}`.

### Status Endpoint

- **Endpoint:** `GET /api/vision/status/{video_id}`
- **Response:** `{"video_id": "...", "stages": {"detect": "done", "track": "pending", ...}}`
- **Status values:** `"pending"` (output file does not exist), `"done"` (output file exists)
- Determined by checking existence of output files on disk. There is no persistent "running" state — if a request is in-flight, the endpoint still reports "pending" until the output file is written.

## Caching

All endpoints follow the existing skip-on-exists pattern: if the output file already exists, return immediately with `skipped: true`. No reprocessing unless the output file is deleted.

## Error Handling

- **GPU service unreachable:** CPU API returns HTTP 502 with the connection error. Timeout for GPU service calls: 600 seconds (video processing is slow).
- **Atomic writes:** GPU services write results to a temporary file (`.tmp` suffix) and rename on completion. This prevents corrupt cache hits if a job fails mid-write.
- **Partial failure:** If a GPU service returns an error, the CPU API forwards the error as HTTP 500 with the GPU service's error message. No output file is written, so retry will reprocess.
- **409 responses:** Include which prerequisite stage(s) are missing (see above).

## Data Exchange

CPU API and GPU services communicate via:
- **HTTP** for triggering work (lightweight JSON payloads: video paths, parameters)
- **Shared filesystem** (`./pipeline_data`) for large artifacts (video files, result JSONs)

No frames or crops are sent over HTTP. GPU services read video from disk and write results to disk.

## Schemas

### CPU API (api/src/schemas/vision.py)

```python
class DetectRequest(BaseModel):
    confidence: float = 0.4
    iou_threshold: float = 0.9
    max_frames: int | None = None

class TrackRequest(BaseModel):
    max_frames: int | None = None

class ClassifyTeamsRequest(BaseModel):
    stride: int = 30
    crop_scale: float = 0.4

class OCRRequest(BaseModel):
    n_consecutive: int = 3
    ocr_interval: int = 5          # run OCR every N frames

class CourtMapRequest(BaseModel):
    keypoint_confidence: float = 0.3
    anchor_confidence: float = 0.5

class DetectResponse(BaseModel):
    video_id: str
    n_frames: int
    n_detections: int
    output_path: str
    skipped: bool = False

class TrackResponse(BaseModel):
    video_id: str
    n_frames: int
    n_tracks: int
    output_path: str
    skipped: bool = False

class ClassifyTeamsResponse(BaseModel):
    video_id: str
    teams: dict[str, str]
    output_path: str
    skipped: bool = False

class OCRResponse(BaseModel):
    video_id: str
    players: dict[str, str]
    output_path: str
    skipped: bool = False

class CourtMapResponse(BaseModel):
    video_id: str
    n_frames_mapped: int
    output_path: str
    skipped: bool = False

class RenderResponse(BaseModel):
    video_id: str
    video_path: str
    skipped: bool = False

class PipelineStatusResponse(BaseModel):
    video_id: str
    stages: dict[str, str]         # values: "pending" | "done"
```

### GPU Service Schemas (internal)

Minimal request/response. Each endpoint receives:
```python
class InferenceRequest(BaseModel):
    video_path: str
    output_dir: str
    params: dict = {}
```

Returns:
```python
class InferenceResponse(BaseModel):
    status: str  # "ok" or "error"
    output_path: str
    error: str | None = None
```

## Configuration

### CPU API (FW_ prefix)

The CPU API retains the `FW_` env prefix from the inherited foreign-whispers codebase. This prefix is kept for backward compatibility with the existing dubbing pipeline settings. New vision-specific settings are added under the same prefix.

New settings in `api/src/core/config.py`:

| Setting | Env Var | Default |
|---|---|---|
| `inference_roboflow_url` | `FW_INFERENCE_ROBOFLOW_URL` | `http://localhost:8091` |
| `inference_vision_url` | `FW_INFERENCE_VISION_URL` | `http://localhost:8092` |
| `analysis_dir` | computed | `data_dir / "analysis"` |

### inference-roboflow (BT_ prefix)

GPU inference services use the `BT_` (BasketTube) prefix to distinguish from the CPU API's `FW_` settings.

| Env Var | Purpose | Default |
|---|---|---|
| `ROBOFLOW_API_KEY` | API key | required |
| `INFERENCE_MODE` | `local` or `remote` | `local` |
| `BT_DATA_DIR` | Pipeline data root | `/app/pipeline_data` |

### inference-vision (BT_ prefix)

| Env Var | Purpose | Default |
|---|---|---|
| `SAM2_REPO` | SAM2 install path | `/opt/segment-anything-2-real-time` |
| `HF_TOKEN` | SigLIP model downloads | required |
| `BT_DATA_DIR` | Pipeline data root | `/app/pipeline_data` |

## GPU Sharing

On a single-GPU host, both `inference-roboflow` and `inference-vision` share the same GPU. They are not expected to run simultaneously during normal pipeline execution — the CPU API calls them sequentially (detect first via roboflow, then track via vision, etc.). If both are idle, their memory footprint is minimal. For multi-GPU hosts, `NVIDIA_VISIBLE_DEVICES` can be set per container to pin each to a specific GPU.

## Docker

### Dockerfiles

The current `Dockerfile` (a GPU/notebook image based on `pytorch`) will be replaced. Three new Dockerfiles will be created:

| File | Base Image | Purpose |
|---|---|---|
| `Dockerfile` | `python:3.11-slim` | CPU API — lightweight orchestrator with FastAPI, no GPU deps |
| `Dockerfile.roboflow` | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` | Roboflow inference service — `inference-gpu`, `supervision`, `sports` |
| `Dockerfile.vision` | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` | SAM2 + TeamClassifier — SAM2 clone/build/checkpoints, `sports` |

The `notebook` service reuses `Dockerfile.vision` with a different CMD (`jupyter lab`). The notebook dependency group from `pyproject.toml` (`jupyterlab`, `ipywidgets`) is installed in `Dockerfile.vision`.

### Docker Compose

All services use `network_mode: host` for simplicity on single-host deployments. With host networking, port mappings are implicit (each service binds directly to its port).

```yaml
services:
  api:
    build: { dockerfile: Dockerfile }
    profiles: [nvidia, cpu]
    network_mode: host
    volumes: [./pipeline_data:/app/pipeline_data]
    # Binds to :8080

  inference-roboflow:
    build: { dockerfile: Dockerfile.roboflow }
    profiles: [nvidia]
    network_mode: host
    shm_size: "8gb"
    GPU reservation
    volumes: [./pipeline_data:/app/pipeline_data]
    # Binds to :8091

  inference-vision:
    build: { dockerfile: Dockerfile.vision }
    profiles: [nvidia]
    network_mode: host
    shm_size: "8gb"
    GPU reservation
    volumes: [./pipeline_data:/app/pipeline_data]
    # Binds to :8092

  notebook:
    build: { dockerfile: Dockerfile.vision }
    profiles: [nvidia]
    network_mode: host
    shm_size: "8gb"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", ...]
    GPU reservation
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./notebooks:/workspace/notebooks
    # Binds to :8888
```

## New Code

| File | Purpose |
|---|---|
| `inference_roboflow/__init__.py` | Package init |
| `inference_roboflow/main.py` | FastAPI app with /api/detect, /api/keypoints, /api/ocr |
| `inference_roboflow/models.py` | Model loading and inference wrappers |
| `inference_vision/__init__.py` | Package init |
| `inference_vision/main.py` | FastAPI app with /api/track, /api/classify-teams |
| `inference_vision/tracker.py` | SAM2Tracker class (extracted from notebook) |
| `inference_vision/classifier.py` | TeamClassifier wrapper |
| `api/src/routers/vision.py` | CPU API vision pipeline router (6 stage endpoints + status) |
| `api/src/schemas/vision.py` | Per-stage request/response Pydantic models |
| `api/src/services/vision_service.py` | HTTP client to GPU services (httpx async) |
| `Dockerfile.roboflow` | GPU Dockerfile for Roboflow inference |
| `Dockerfile.vision` | GPU Dockerfile for SAM2 + TeamClassifier + notebook |

## Existing Code Unchanged

All foreign-whispers routers (download, transcribe, translate, tts, stitch, eval), services, and inference backends remain functional for the commentary analysis side of BasketTube.
