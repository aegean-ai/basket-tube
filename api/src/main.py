"""BasketTube FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.config import settings

logger = logging.getLogger(__name__)

# ── Logfire instrumentation ───────────────────────────────────────────
# Configure before app creation so all spans are captured.
# Uses the project token from .logfire/ (set via `logfire projects use basket-tube`).
# Falls back to FW_LOGFIRE_WRITE_TOKEN env var, or disables if neither is set.
try:
    import logfire
    _token = settings.logfire_write_token or None
    if _token:
        logfire.configure(token=_token, service_name="basket-tube-api")
        _logfire_available = True
        logger.info("Logfire tracing enabled.")
    else:
        _logfire_available = False
        logger.info("Logfire disabled — no FW_LOGFIRE_WRITE_TOKEN set.")
except ImportError:
    _logfire_available = False
    logger.info("Logfire not installed — tracing disabled.")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title=settings.app_title,
    )

    # Instrument FastAPI with Logfire
    if _logfire_available:
        logfire.instrument_fastapi(app)

    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from api.src.routers.download import router as download_router
    from api.src.routers.transcribe import router as transcribe_router
    from api.src.routers.vision import router as vision_router
    from api.src.routers.captions import router as captions_router
    from api.src.routers.settings import router as settings_router
    from api.src.routers.pipeline import router as pipeline_router

    app.include_router(download_router)
    app.include_router(transcribe_router)
    app.include_router(vision_router)
    app.include_router(captions_router)
    app.include_router(settings_router)
    app.include_router(pipeline_router)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/videos")
    async def list_videos():
        from api.src.video_registry import get_all_videos
        return [
            {"id": v.id, "title": v.title, "url": v.url}
            for v in get_all_videos()
        ]

    return app


app = create_app()
