"""Whisper STT service — HTTP client to the remote Whisper container."""

import logging

import requests

from api.src.config import settings

logger = logging.getLogger(__name__)


def transcribe(audio_path: str) -> dict:
    """POST audio/video to the remote Whisper service and return a result dict.

    The remote service (speaches) exposes an OpenAI-compatible endpoint at
    ``{whisper_api_url}/v1/audio/transcriptions``.

    Returns dict with "text", "language", and "segments" keys.
    """
    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    logger.info("Remote Whisper transcription: POST %s", url)

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (audio_path, f, "audio/wav")},
            data={
                "model": settings.whisper_model,
                "response_format": "verbose_json",
            },
            timeout=300,
        )

    response.raise_for_status()
    return response.json()
