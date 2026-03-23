"""Pydantic schemas for the basketball vision pipeline API contract."""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    model_id: str = "basketball-player-detection-3-ycjdo/4"
    confidence: float = 0.4
    iou_threshold: float = 0.9
    max_frames: int | None = None


class TrackRequest(BaseModel):
    det_config_key: str
    max_frames: int | None = None
    sam2_checkpoint: str = "sam2.1_hiera_large.pt"
    iou_threshold: float = 0.5
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30


class ClassifyTeamsRequest(BaseModel):
    det_config_key: str
    stride: int = 30
    crop_scale: float = 0.4
    embedding_model: str = "google/siglip-base-patch16-224"
    n_teams: int = 2


class OCRRequest(BaseModel):
    track_config_key: str
    model_id: str = "basketball-jersey-numbers-ocr/3"
    n_consecutive: int = 3
    ocr_interval: int = 5


class CourtMapRequest(BaseModel):
    det_config_key: str
    model_id: str = "basketball-court-detection-2/14"
    keypoint_confidence: float = 0.3
    anchor_confidence: float = 0.5


class RenderRequest(BaseModel):
    det_config_key: str
    track_config_key: str
    teams_config_key: str
    jerseys_config_key: str
    court_config_key: str | None = None  # optional — render works without court


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DetectResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames: int
    n_detections: int
    skipped: bool = False


class TrackResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames: int
    n_tracks: int
    skipped: bool = False


class ClassifyTeamsResponse(BaseModel):
    video_id: str
    config_key: str
    palette: dict[str, dict]
    skipped: bool = False


class OCRResponse(BaseModel):
    video_id: str
    config_key: str
    players: dict[str, str]
    skipped: bool = False


class CourtMapResponse(BaseModel):
    video_id: str
    config_key: str
    n_frames_mapped: int
    skipped: bool = False


class RenderResponse(BaseModel):
    video_id: str
    config_key: str
    skipped: bool = False


class StageAcceptedResponse(BaseModel):
    stage: str
    config_key: str
    sse_url: str


class StageStatusResponse(BaseModel):
    status: str
    config_key: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error: str | None = None


class PipelineStatusResponse(BaseModel):
    video_id: str
    stages: dict[str, StageStatusResponse]


# ---------------------------------------------------------------------------
# GPU service schemas (internal)
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None
