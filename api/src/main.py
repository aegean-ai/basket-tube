"""BasketTube FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — lazy model loading on first use."""
    app.state._whisper_model = None
    logger.info("Application ready (models will load on first use).")

    # Configure Logfire if a write token is available
    if settings.logfire_write_token:
        try:
            import logfire
            logfire.configure(
                write_token=settings.logfire_write_token,
                service_name="basket-tube",
            )
            logfire.instrument_fastapi(app)
            logger.info("Logfire tracing enabled.")
        except ImportError:
            logger.info("Logfire not installed — tracing disabled.")

    yield

    if app.state._whisper_model is not None:
        del app.state._whisper_model
    logger.info("Models unloaded.")


def get_whisper_model(app):
    """Lazy-load Whisper model on first use."""
    if app.state._whisper_model is None:
        logger.info("Loading Whisper model (%s)...", settings.whisper_model)
        import whisper
        app.state._whisper_model = whisper.load_model(settings.whisper_model)
        logger.info("Whisper model loaded.")
    return app.state._whisper_model


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title=settings.app_title,
        lifespan=lifespan,
    )

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

    app.include_router(download_router)
    app.include_router(transcribe_router)
    app.include_router(vision_router)
    app.include_router(captions_router)

    @app.get("/healthz")
    async def healthz():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/api/videos")
    async def list_videos():
        """Return the video catalog from video_registry.yml."""
        from api.src.core.video_registry import get_all_videos
        return [
            {"id": v.id, "title": v.title, "url": v.url}
            for v in get_all_videos()
        ]

    return app


app = create_app()
