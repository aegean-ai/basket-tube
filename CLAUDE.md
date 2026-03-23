# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**BasketTube** — an aegean.ai tech demonstrator for AI-powered basketball game analysis. It combines computer vision, speech-to-text, and vision-language models to analyze basketball games through two complementary approaches:

1. **Commentary Analysis** — extract player performance insights from game audio via STT + NLP
2. **Video Verification** — detect, track, and identify players directly from footage via CV models

### Key Capabilities

- **Player Detection & Tracking** — RF-DETR object detection, SAM2 segmentation, ByteTrack tracking
- **Player Identification** — jersey number OCR, team classification via color clustering
- **Play Chunking** — automatic segmentation into offensive/defensive plays (pick-and-roll, iso, zone, etc.)
- **Action Recognition** — shooting, passing, dribbling, rebounding, defending, assisting
- **Court Mapping** — keypoint detection + homography for bird's-eye spatial visualization
- **Commentary Analysis** — STT transcription + VLM-powered natural language querying

```text
basket-tube/
├── api/src/                     # CPU API (FastAPI, async)
│   ├── main.py                  # App factory (create_app)
│   ├── config.py                # Pydantic settings (FW_ env prefix)
│   ├── artifacts.py             # Config keys, paths, sidecars, deletion
│   ├── routers/
│   │   ├── pipeline.py          # Pipeline run/cancel/SSE/staleness
│   │   ├── vision.py            # 6 vision stage endpoints + DELETE
│   │   ├── download.py          # Video download (async via to_thread)
│   │   ├── transcribe.py        # Whisper STT (async httpx)
│   │   ├── captions.py          # Text timeline
│   │   └── settings.py          # Per-video settings with migration
│   ├── schemas/                 # Pydantic request/response models
│   └── services/
│       ├── pipeline_orchestrator.py  # Async DAG scheduler + SSE broadcast
│       ├── event_bus.py              # Broadcast event bus
│       ├── vision_service.py         # GPU service HTTP client
│       └── whisper_service.py        # Async Whisper HTTP client
├── basket_tube/inference/       # GPU inference service
│   ├── main.py                  # FastAPI app (5 endpoints + progress)
│   └── progress.py              # Atomic _progress.json writer
├── frontend/                    # Next.js 16 + shadcn/ui
├── notebooks/                   # Jupyter notebooks
├── pipeline_data/               # Runtime artifacts (videos, outputs)
├── Dockerfile.api               # CPU API image (non-root appuser)
├── Dockerfile.gpu               # GPU inference image (non-root appuser)
├── docker-compose.yml           # 4 services (api, inference, frontend, whisper)
├── pyproject.toml               # uv-managed dependencies
└── uv.lock                      # Locked dependency versions
```

## Running the App

**Always use Docker Compose — never run notebooks on bare metal.**

This host has an NVIDIA GPU; always use the `nvidia` profile:

```bash
# Set API keys first
cp .env.example .env
# Edit .env with your HF_TOKEN and ROBOFLOW_API_KEY

# Build and start
docker compose --profile nvidia up -d

# Open the app
# Frontend: http://localhost:8501
# API:      http://localhost:8080
```

After changing `pyproject.toml` or the `Dockerfile`, rebuild:

```bash
docker compose --profile nvidia build
docker compose --profile nvidia up -d
```

To stop:

```bash
docker compose --profile nvidia down
```

## Architecture

### Async Pipeline

The CPU API runs a fully async pipeline orchestrator with dependency-aware parallelism:

```text
detect ─┬→ track → ocr
        ├→ classify-teams        (all 3 run concurrently after detect)
        └→ court-map
```

- Pipeline triggered via `POST /api/pipeline/run/{video_id}` (returns 202 immediately)
- Real-time progress via SSE (`GET /api/pipeline/events/{video_id}`)
- Broadcast EventBus supports multiple tabs + browser refresh reconnect
- GPU service writes atomic `_progress.json` for frame-level progress
- Per-stage settings with staleness detection (amber "Outdated" badge)
- Per-stage Run/Re-run buttons (disabled while pipeline is active)

### Key dependencies

- **PyTorch 2.7 + CUDA 12.8** — base Docker image provides GPU compute
- **Roboflow `inference-gpu`** — player detection, keypoint models, OCR via Roboflow
- **`supervision`** — annotation, tracking, and video utilities
- **`sports`** (roboflow, `feat/basketball` branch) — court configuration, team classification, view transforms
- **`logfire`** — observability (traces pipeline.run, gpu.{stage}, whisper.transcribe)
- **`httpx`** — async HTTP for Whisper and GPU service calls
- **Next.js 16 + shadcn/ui** — SSE-driven frontend with per-stage settings sidebar

### Logfire Observability

Traces and spans are sent to Pydantic Logfire at https://logfire-us.pydantic.dev/pantelis/basket-tube

To debug exceptions using the Logfire MCP server:

```bash
claude "$(uvx logfire@latest --region us prompt --project pantelis/basket-tube fix-span-issue:<SPAN_ID> --claude)"
```

### Environment variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token (model downloads) |
| `ROBOFLOW_API_KEY` | Roboflow API key (inference models) |
| `FW_INFERENCE_GPU_URL` | GPU service URL (default: `http://localhost:8090`) |
| `FW_WHISPER_API_URL` | Whisper service URL (default: `http://localhost:8000`) |
| `FW_LOGFIRE_WRITE_TOKEN` | Pydantic Logfire write token |
| `ONNXRUNTIME_EXECUTION_PROVIDERS` | Set to `[CUDAExecutionProvider]` for GPU inference |

### Design decisions

- **Async everywhere** — `asyncio.to_thread()` for yt_dlp, `httpx.AsyncClient` for Whisper, async httpx for GPU calls. No blocking on the FastAPI event loop.
- **SSE for progress** — Server-Sent Events with broadcast EventBus (append-only log + per-subscriber cursors). No WebSocket needed for one-directional server→client flow.
- **Config-key caching** — each parameter combination produces a unique `c-{hash}` directory. Changing confidence or swapping models never returns stale results.
- **Staleness detection** — backend computes expected config keys from current settings and compares against cached artifacts. Frontend shows "Outdated" badge.
- **Non-root containers** — Both API and GPU containers run as `appuser` (UID/GID from host) to avoid permission issues with mounted volumes.
- **Per-stage settings** — each pipeline stage has its own settings panel with model configuration, following the Foreign Whispers sidebar pattern.
- `pyproject.toml` manages all Python deps via `uv`; the `sports` package is sourced from a git branch.
