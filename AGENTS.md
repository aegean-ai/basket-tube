# Agent Instructions

## Project Context

**BasketTube** is an aegean.ai tech demonstrator for AI-powered basketball game analysis. It combines:

- **Computer vision** — player detection (RF-DETR), segmentation/tracking (SAM2), team classification, jersey OCR, court mapping
- **Speech-to-text** — commentary transcription and semantic parsing
- **Vision-language models** — SmolVLM2, SigLIP for action recognition and play classification

The project has two main surfaces:
1. **Jupyter notebooks** (`notebooks/`) — CV pipeline prototyping, runs in a GPU Docker container
2. **Web application** — FastAPI backend (`api/`) + Next.js/shadcn frontend (`frontend/`) for the full interactive experience

## Development Workflow

### Notebooks (GPU)

```bash
docker compose --profile nvidia up -d
# JupyterLab at http://localhost:8888
```

Source notebooks live in `notebooks/`, mounted into the container at `/workspace/notebooks`.

### Backend / Frontend

The FastAPI API (`api/src/`) and Next.js frontend (`frontend/`) are retained from the inherited codebase and will be adapted for BasketTube's needs.

## Key Commands

```bash
# Build and start notebook container
docker compose --profile nvidia build
docker compose --profile nvidia up -d

# View logs
docker compose --profile nvidia logs -f notebook

# Stop
docker compose --profile nvidia down
```

## Environment Setup

Copy `.env.example` to `.env` and set:
- `HF_TOKEN` — Hugging Face API token
- `ROBOFLOW_API_KEY` — Roboflow API key

## Dependencies

Managed via `pyproject.toml` with `uv`. To add a package:
1. Add it to `pyproject.toml`
2. Run `uv lock`
3. Rebuild the Docker image
