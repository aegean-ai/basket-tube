"""Whisper STT service — HTTP client to the remote Whisper container."""

import logging
import os

import requests

from api.src.config import settings

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

logger = logging.getLogger(__name__)


def transcribe(audio_path: str) -> dict:
    """POST audio/video to the remote Whisper service and return a result dict."""
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
    mime = mime_types.get(ext, "application/octet-stream")

    logger.info("Remote Whisper transcription: POST %s (file=%s, model=%s)", url, filename, settings.whisper_model)

    ctx = logfire.span("whisper.transcribe", filename=filename, model=settings.whisper_model) if logfire else None
    if ctx:
        ctx.__enter__()

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": (filename, f, mime)},
                data={"model": settings.whisper_model, "response_format": "verbose_json"},
                timeout=600,
            )

        if not response.ok:
            logger.error("Whisper service returned %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()
        result = response.json()

        if logfire:
            logfire.info("whisper.transcribe complete",
                         segments=len(result.get("segments", [])),
                         language=result.get("language", ""))
        return result
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
