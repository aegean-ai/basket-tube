"""Settings persistence Pydantic models — stage-keyed structure."""
from pydantic import BaseModel


class TeamInfo(BaseModel):
    name: str = "Team A"
    color: str = "#006BB6"


class GameContext(BaseModel):
    teams: dict[str, TeamInfo] = {
        "0": TeamInfo(name="Team A", color="#006BB6"),
        "1": TeamInfo(name="Team B", color="#007A33"),
    }
    roster: dict[str, str] = {}


class TranscribeSettings(BaseModel):
    model: str = "Systran/faster-whisper-medium"
    use_youtube_captions: bool = True


class DetectSettings(BaseModel):
    model_id: str = "basketball-player-detection-3-ycjdo/4"
    confidence: float = 0.4
    iou_threshold: float = 0.9


class TrackSettings(BaseModel):
    iou_threshold: float = 0.5
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30


class OCRSettings(BaseModel):
    model_id: str = "basketball-jersey-numbers-ocr/3"
    ocr_interval: int = 5
    n_consecutive: int = 3


class TeamsSettings(BaseModel):
    embedding_model: str = "google/siglip-base-patch16-224"
    n_teams: int = 2
    crop_scale: float = 0.4
    stride: int = 30


class CourtMapSettings(BaseModel):
    model_id: str = "basketball-court-detection-2/14"
    keypoint_confidence: float = 0.3
    anchor_confidence: float = 0.5


class StageSettings(BaseModel):
    transcribe: TranscribeSettings = TranscribeSettings()
    detect: DetectSettings = DetectSettings()
    track: TrackSettings = TrackSettings()
    ocr: OCRSettings = OCRSettings()
    teams: TeamsSettings = TeamsSettings()
    court_map: CourtMapSettings = CourtMapSettings()


class AnalysisSettings(BaseModel):
    game_context: GameContext = GameContext()
    stages: StageSettings = StageSettings()


# --- Migration from old flat format ---

class _OldAdvanced(BaseModel):
    confidence: float = 0.4
    iou_threshold: float = 0.9
    ocr_interval: int = 5
    crop_scale: float = 0.4
    stride: int = 30


def migrate_settings(data: dict) -> AnalysisSettings:
    """Accept either old flat format or new stage-keyed format."""
    if "stages" in data:
        return AnalysisSettings(**data)

    gc = data.get("game_context", {})
    old = _OldAdvanced(**(data.get("advanced", {})))

    stages = StageSettings(
        detect=DetectSettings(confidence=old.confidence, iou_threshold=old.iou_threshold),
        ocr=OCRSettings(ocr_interval=old.ocr_interval),
        teams=TeamsSettings(crop_scale=old.crop_scale, stride=old.stride),
    )
    return AnalysisSettings(game_context=GameContext(**gc), stages=stages)
