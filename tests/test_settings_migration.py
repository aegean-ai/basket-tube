from api.src.schemas.settings import AnalysisSettings, migrate_settings


def test_new_stage_keyed_settings():
    s = AnalysisSettings()
    assert s.stages.detect.confidence == 0.4
    assert s.stages.transcribe.model == "Systran/faster-whisper-medium"
    assert s.stages.ocr.ocr_interval == 5


def test_migrate_old_flat_format():
    old = {
        "game_context": {"teams": {"0": {"name": "A", "color": "#000"}}, "roster": {}},
        "advanced": {"confidence": 0.7, "iou_threshold": 0.8, "ocr_interval": 10, "crop_scale": 0.3, "stride": 15},
    }
    result = migrate_settings(old)
    assert result.stages.detect.confidence == 0.7
    assert result.stages.detect.iou_threshold == 0.8
    assert result.stages.ocr.ocr_interval == 10
    assert result.stages.teams.crop_scale == 0.3
    assert result.stages.teams.stride == 15
    assert result.game_context.teams["0"].name == "A"


def test_new_format_passes_through():
    new = {
        "game_context": {"teams": {}, "roster": {}},
        "stages": {
            "detect": {"model_id": "custom/1", "confidence": 0.5, "iou_threshold": 0.85},
            "transcribe": {"model": "tiny", "use_youtube_captions": False},
        },
    }
    result = migrate_settings(new)
    assert result.stages.detect.model_id == "custom/1"
    assert result.stages.transcribe.use_youtube_captions is False


def test_empty_dict_uses_defaults():
    result = migrate_settings({})
    assert result.stages.detect.confidence == 0.4
    assert result.game_context.roster == {}
