"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_title: str = "BasketTube API"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # CORS
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]

    # STT configuration
    whisper_model: str = "Systran/faster-whisper-medium"
    whisper_api_url: str = "http://localhost:8000"
    stt_model_dir: str = "whisper"

    # File paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "pipeline_data" / "api"

    # ── pipeline directory layout ───────────────────────────────────────

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def youtube_captions_dir(self) -> Path:
        return self.data_dir / "youtube_captions"

    @property
    def transcriptions_dir(self) -> Path:
        return self.data_dir / "transcriptions" / self.stt_model_dir

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def settings_dir(self) -> Path:
        return self.data_dir / "settings"

    # GPU inference service URL
    inference_gpu_url: str = "http://localhost:8090"

    # S3 storage (optional)
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""

    # HuggingFace token
    hf_token: str = ""

    # Logfire write token
    logfire_write_token: str = ""

    model_config = {"env_prefix": "FW_"}


settings = Settings()
