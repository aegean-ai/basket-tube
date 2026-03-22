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
├── notebooks/                   # Jupyter notebooks (run inside Docker with GPU)
│   └── basketball_ai_how_to_detect_track_and_identify_basketball_players.ipynb
├── api/src/                     # Layered FastAPI backend
│   ├── main.py                  # App factory (create_app)
│   ├── core/config.py           # Pydantic settings (env-driven)
│   ├── core/dependencies.py     # FastAPI Depends providers
│   ├── routers/                 # Route modules
│   ├── schemas/                 # Pydantic request/response models
│   ├── services/                # Business logic
│   └── inference/               # Model backend abstraction
├── frontend/                    # Next.js + shadcn/ui frontend
├── pipeline_data/               # Runtime artifacts (videos, outputs)
├── Dockerfile                   # GPU-enabled Jupyter environment (PyTorch + CUDA 12.8)
├── docker-compose.yml           # Notebook service with nvidia profile
├── pyproject.toml               # uv-managed dependencies
└── uv.lock                     # Locked dependency versions
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

# Open JupyterLab
# http://localhost:8888
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

### CV Pipeline (notebook)

```text
Video frames → RF-DETR detection → SAM2 segmentation/tracking → Team classification → Jersey OCR → Court homography → Bird's-eye mapping
```

### Key dependencies

- **PyTorch 2.7 + CUDA 12.8** — base Docker image provides GPU compute
- **SAM2** (`segment-anything-2-real-time`) — pre-installed at `/opt/segment-anything-2-real-time` with checkpoints
- **Roboflow `inference-gpu`** — player detection and keypoint models via Roboflow API
- **`supervision`** — annotation, tracking, and video utilities
- **`sports`** (roboflow, `feat/basketball` branch) — court configuration, team classification, view transforms
- **`logfire`** — optional observability

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
| `SAM2_REPO` | Path to SAM2 install (default: `/opt/segment-anything-2-real-time`) |
| `ONNXRUNTIME_EXECUTION_PROVIDERS` | Set to `[CUDAExecutionProvider]` for GPU inference |

### Design decisions

- All heavy dependencies (SAM2, checkpoints, Python packages) are pre-installed in the Docker image.
- The detection notebook is adapted from a Google Colab original; Colab-specific code has been replaced with `os.environ`.
- `pyproject.toml` manages all Python deps via `uv`; the `sports` package is sourced from a git branch.
- FastAPI backend and Next.js frontend are retained for the full application (commentary analysis, video player, chat interface).
