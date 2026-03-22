"""POST /api/transcribe/{video_id} — Whisper transcription."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Query

from api.src.config import settings
from api.src.dependencies import resolve_title
from api.src.schemas.transcribe import TranscribeResponse
from api.src.services import whisper_service

router = APIRouter(prefix="/api")


def _youtube_captions_to_segments(caption_path: pathlib.Path) -> dict:
    """Convert YouTube line-delimited JSON captions to Whisper-compatible result dict."""
    segments = []
    full_text_parts = []
    for i, line in enumerate(caption_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        seg = json.loads(line)
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        duration = seg.get("duration", 0)
        if not text or duration <= 0:
            continue
        segments.append({
            "id": i,
            "start": start,
            "end": start + duration,
            "text": text,
        })
        full_text_parts.append(text)
    return {
        "language": "en",
        "text": " ".join(full_text_parts),
        "segments": segments,
    }


@router.post("/transcribe/{video_id}", response_model=TranscribeResponse)
async def transcribe_endpoint(
    video_id: str,
    use_youtube_captions: bool = Query(True, description="Use YouTube captions when available, skipping Whisper"),
):
    """Transcribe video audio via remote Whisper service.

    When use_youtube_captions is True (default), YouTube captions are used if
    available, skipping Whisper entirely. When False, Whisper always runs.
    """
    transcriptions_dir = settings.transcriptions_dir
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in registry")

    transcript_path = transcriptions_dir / f"{title}.json"

    # Return cached result if it exists
    if transcript_path.exists() and use_youtube_captions:
        data = json.loads(transcript_path.read_text())
        return TranscribeResponse(
            video_id=video_id,
            language=data.get("language", "en"),
            text=data.get("text", ""),
            segments=data.get("segments", []),
            skipped=True,
        )

    # Prefer YouTube captions over running Whisper
    if use_youtube_captions:
        yt_caption_path = settings.youtube_captions_dir / f"{title}.txt"
        if yt_caption_path.exists():
            result = _youtube_captions_to_segments(yt_caption_path)
            transcript_path.write_text(json.dumps(result))
            return TranscribeResponse(
                video_id=video_id,
                language=result["language"],
                text=result["text"],
                segments=result["segments"],
                skipped=True,
            )

    # Run Whisper STT via remote service
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {title}.mp4")

    result = whisper_service.transcribe(str(video_path))
    transcript_path.write_text(json.dumps(result))

    return TranscribeResponse(
        video_id=video_id,
        language=result.get("language", "en"),
        text=result.get("text", ""),
        segments=result.get("segments", []),
    )
