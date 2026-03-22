"""Whisper STT service — HTTP client to the remote Whisper container."""

import logging
import os

import requests

from api.src.config import settings

logger = logging.getLogger(__name__)

# Map file extensions to MIME types
_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".webm": "video/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def transcribe(audio_path: str) -> dict:
    """POST audio/video to the remote Whisper service and return a result dict.

    The remote service (speaches) exposes an OpenAI-compatible endpoint at
    ``{whisper_api_url}/v1/audio/transcriptions``.

    Returns dict with "text", "language", and "segments" keys.
    """
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime = _MIME_TYPES.get(ext, "application/octet-stream")

    logger.info("Remote Whisper transcription: POST %s (file=%s, mime=%s, model=%s)",
                url, filename, mime, settings.whisper_model)

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (filename, f, mime)},
            data={
                "model": settings.whisper_model,
                "response_format": "verbose_json",
            },
            timeout=300,
        )

    if not response.ok:
        logger.error("Whisper service returned %s: %s", response.status_code, response.text[:500])
    response.raise_for_status()
    return response.json()
