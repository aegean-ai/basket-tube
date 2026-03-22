"""POST /api/download — download YouTube video + captions (issue by5)."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Request

from api.src.config import settings
from api.src.video_registry import get_video
from api.src.schemas.download import CaptionSegment, DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments."""
    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is a registered local video (no YouTube download needed)
    entry = get_video(body.url)
    if entry and entry.url == "local":
        stem = entry.title
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.exists():
            raise HTTPException(404, f"Local video file not found: {stem}.mp4")
        caption_path = captions_dir / f"{stem}.txt"
        segments = _download_service.read_caption_segments(caption_path) if caption_path.exists() else []
        return DownloadResponse(video_id=entry.id, title=stem, caption_segments=segments)

    # YouTube download
    video_id, title = _download_service.get_video_info(body.url)
    entry = get_video(video_id)
    stem = entry.title if entry else title.replace(":", "")

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    if not video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), stem)

    if not caption_path.exists():
        try:
            _download_service.download_caption(body.url, str(captions_dir), stem)
        except Exception:
            pass  # Captions may be disabled for this video — not a fatal error

    segments = _download_service.read_caption_segments(caption_path) if caption_path.exists() else []

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )
