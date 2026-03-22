"""Captions timeline router — text timeline for dataset builder."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.artifacts import (
    artifact_path,
    atomic_write_json,
    check_stale,
    config_key,
    read_status,
    status_path_for,
    write_status,
)
from api.src.config import settings
from api.src.video_registry import resolve_stem
from api.src.schemas.captions import TextTimelineRequest, TextTimelineResponse
from api.src.services.text_timeline_service import build_timeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/captions", tags=["captions"])


@router.post("/timeline/{video_id}", response_model=TextTimelineResponse)
async def timeline(video_id: str, req: TextTimelineRequest | None = None):
    """Build a normalized text timeline from existing transcription output."""
    req = req or TextTimelineRequest()
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")

    # Check transcription exists
    transcription_path = settings.data_dir / "transcriptions" / req.stt_model_dir / f"{stem}.json"
    if not transcription_path.exists():
        raise HTTPException(
            409,
            detail={"detail": "Transcription must be completed before building timeline", "missing": ["transcribe"]},
        )

    # Determine source type
    yt_caption_path = settings.youtube_captions_dir / f"{stem}.txt"
    source = "caption" if yt_caption_path.exists() else "stt"

    cfg_params = {"stt_model_dir": req.stt_model_dir, "source_type": source, "lexicon_version": req.lexicon_version}
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "text_timeline", cfg_key, stem)
    sidecar = status_path_for(out)

    # Skip if exists
    if out.exists():
        data = json.loads(out.read_text())
        return TextTimelineResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_segments=len(data.get("segments", [])),
            source=data.get("source", source),
            skipped=True,
        )

    # Check not already running (with crash recovery)
    current = check_stale(sidecar, timeout_s=60.0)
    if current["status"] == "active":
        raise HTTPException(409, detail={"detail": "Timeline is already being built"})

    write_status(sidecar, "active")
    try:
        transcript = json.loads(transcription_path.read_text())
        result = build_timeline(
            transcript,
            source=source,
            lexicon_version=req.lexicon_version,
            stt_model_dir=req.stt_model_dir,
        )

        atomic_write_json(out, result)
        write_status(sidecar, "complete", config_key=cfg_key)

        return TextTimelineResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_segments=len(result["segments"]),
            source=source,
        )
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(500, detail=f"Timeline construction failed: {exc}")
