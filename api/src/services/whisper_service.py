"""Whisper STT service — HTTP client to the remote Whisper container."""

import logging
import os
import subprocess
import tempfile

import requests

from api.src.config import settings

logger = logging.getLogger(__name__)


def _extract_audio(video_path: str) -> str:
    """Extract audio from video to a temporary WAV file using ffmpeg."""
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(audio_path):
        return audio_path

    logger.info("Extracting audio: %s → %s", video_path, audio_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path,
        ],
        check=True,
    )
    return audio_path


def transcribe(video_path: str) -> dict:
    """Extract audio from video, POST to remote Whisper, return result dict.

    The remote service (speaches) exposes an OpenAI-compatible endpoint at
    ``{whisper_api_url}/v1/audio/transcriptions``.

    Returns dict with "text", "language", and "segments" keys.
    """
    audio_path = _extract_audio(video_path)

    url = f"{settings.whisper_api_url.rstrip('/')}/v1/audio/transcriptions"
    logger.info("Remote Whisper transcription: POST %s (%s)", url, audio_path)

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={
                "model": settings.whisper_model,
                "response_format": "verbose_json",
            },
            timeout=600,
        )

    response.raise_for_status()
    return response.json()
