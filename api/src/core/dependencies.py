"""FastAPI Depends providers for services and configuration."""

from functools import lru_cache

from api.src.core.config import Settings
from api.src.core.video_registry import resolve_title  # noqa: F401 — re-export


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance for use with FastAPI Depends."""
    return Settings()
