"""Settings persistence Pydantic models."""
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


class AdvancedSettings(BaseModel):
    confidence: float = 0.4
    iou_threshold: float = 0.9
    ocr_interval: int = 5
    crop_scale: float = 0.4
    stride: int = 30


class AnalysisSettings(BaseModel):
    game_context: GameContext = GameContext()
    advanced: AdvancedSettings = AdvancedSettings()
