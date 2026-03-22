"""Pydantic request/response models for the captions timeline endpoint."""

from pydantic import BaseModel


class TextTimelineRequest(BaseModel):
    stt_model_dir: str = "whisper"
    lexicon_version: str = "v0.1"


class TextTimelineResponse(BaseModel):
    video_id: str
    config_key: str
    n_segments: int
    source: str
    skipped: bool = False
