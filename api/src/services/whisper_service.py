"""Whisper STT service — HTTP client to the remote Whisper container."""

import logging
import os

import requests

from api.src.config import settings

logger = logging.getLogger(__name__)


def transcribe(audio_path: str) -> dict:
    """POST audio to the remote Whisper service and return a Whisper-format result dict.

    The remote service (speaches) exposes an OpenAI-compatible endpoint at
    ``{whisper_api_url}/v1/audio/transcriptions``.

    Returns dict with "text", "language", and "segments" keys.
    """
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    logger.info("Remote Whisper transcription: POST %s (%s)", url, audio_path)

    filename = os.path.basename(audio_path)
    # Determine MIME type from extension
    mime = "video/mp4" if filename.endswith(".mp4") else "audio/wav"

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (filename, f, mime)},
            data={
                "model": settings.whisper_model,
                "response_format": "verbose_json",
            },
            timeout=600,
        )

    response.raise_for_status()
    return response.json()
