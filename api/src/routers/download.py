"""POST /api/download — download YouTube video + captions (issue by5)."""

import asyncio
import json
import pathlib

from fastapi import APIRouter, HTTPException, Request

from api.src.config import settings
from api.src.video_registry import get_video
from api.src.schemas.download import CaptionSegment, DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

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
    entry = get_video(body.url)  # lookup by video ID
    if entry and entry.url == "local":
        stem = entry.title
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.exists():
            raise HTTPException(404, f"Local video file not found: {stem}.mp4")
        caption_path = captions_dir / f"{stem}.txt"
        segments = _download_service.read_caption_segments(caption_path) if caption_path.exists() else []
        with (logfire.span("pipeline.download", video_id=entry.id, source="local") if logfire else _null_ctx()):
            return DownloadResponse(video_id=entry.id, title=stem, caption_segments=segments)

    # YouTube download
    video_id, title = await asyncio.to_thread(_download_service.get_video_info, body.url)
    entry = get_video(video_id)
    stem = entry.title if entry else title.replace(":", "")

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    async def _do_youtube_download():
        if not video_path.exists():
            await asyncio.to_thread(_download_service.download_video, body.url, str(videos_dir), stem)

        if not caption_path.exists():
            try:
                await asyncio.to_thread(_download_service.download_caption, body.url, str(captions_dir), stem)
            except Exception:
                pass  # Captions may be disabled for this video — not a fatal error

        segments = _download_service.read_caption_segments(caption_path) if caption_path.exists() else []
        return segments

    if logfire:
        with logfire.span("pipeline.download", video_id=video_id, source="youtube"):
            segments = await _do_youtube_download()
    else:
        segments = await _do_youtube_download()

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )


class _null_ctx:
    """No-op context manager used when logfire is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
