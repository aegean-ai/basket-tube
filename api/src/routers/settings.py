"""Settings persistence — GET/PUT per-video analysis settings."""
import json
import logging
from fastapi import APIRouter
from api.src.artifacts import atomic_write_json
from api.src.config import settings
from api.src.schemas.settings import AnalysisSettings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["settings"])


def _settings_path(video_id: str):
    return settings.settings_dir / f"{video_id}.json"


@router.get("/settings/{video_id}", response_model=AnalysisSettings)
async def get_settings(video_id: str):
    path = _settings_path(video_id)
    if path.exists():
        data = json.loads(path.read_text())
        return AnalysisSettings(**data)
    return AnalysisSettings()


@router.put("/settings/{video_id}", response_model=AnalysisSettings)
async def put_settings(video_id: str, body: AnalysisSettings):
    path = _settings_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, body.model_dump())
    return body
