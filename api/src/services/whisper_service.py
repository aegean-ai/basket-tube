"""Whisper STT service — async HTTP client to the remote Whisper container."""

import logging
import os
from contextlib import nullcontext

import httpx

from api.src.config import settings

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

logger = logging.getLogger(__name__)


async def transcribe(audio_path: str, model: str | None = None) -> dict:
    """POST audio/video to the remote Whisper service and return a result dict."""
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    model = model or settings.whisper_model
    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
    mime = mime_types.get(ext, "application/octet-stream")

    logger.info("Remote Whisper transcription: POST %s (file=%s, model=%s)", url, filename, model)

    span_ctx = logfire.span("whisper.transcribe", filename=filename, model=model) if logfire else nullcontext()

    with span_ctx:
        async with httpx.AsyncClient(timeout=600) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    url,
                    files={"file": (filename, f, mime)},
                    data={"model": model, "response_format": "verbose_json"},
                )

        if not response.is_success:
            logger.error("Whisper service returned %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()
        result = response.json()

        if logfire:
            logfire.info("whisper.transcribe complete",
                         segments=len(result.get("segments", [])),
                         language=result.get("language", ""))
        return result
