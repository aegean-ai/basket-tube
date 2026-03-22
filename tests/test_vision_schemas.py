"""Tests for basketball vision pipeline Pydantic schemas."""

import pytest

from api.src.schemas.vision import (
    ClassifyTeamsRequest,
    ClassifyTeamsResponse,
    CourtMapRequest,
    DetectRequest,
    DetectResponse,
    InferenceRequest,
    InferenceResponse,
    OCRRequest,
    PipelineStatusResponse,
    RenderRequest,
    StageStatusResponse,
    TrackRequest,
    TrackResponse,
)


class TestRequestDefaults:
    def test_detect_request_defaults(self):
        req = DetectRequest()
        assert req.model_id == "basketball-player-detection-3-ycjdo/4"
        assert req.confidence == 0.4
        assert req.iou_threshold == 0.9
        assert req.max_frames is None

    def test_track_request_requires_det_config_key(self):
        with pytest.raises(Exception):
            TrackRequest()  # det_config_key is required

    def test_track_request_defaults(self):
        req = TrackRequest(det_config_key="det-abc123")
        assert req.sam2_checkpoint == "sam2.1_hiera_large.pt"
        assert req.max_frames is None
        assert req.det_config_key == "det-abc123"

    def test_classify_teams_request_defaults(self):
        req = ClassifyTeamsRequest(det_config_key="det-abc123")
        assert req.stride == 30
        assert req.crop_scale == 0.4
        assert req.det_config_key == "det-abc123"

    def test_ocr_request_requires_track_config_key(self):
        with pytest.raises(Exception):
            OCRRequest()  # track_config_key is required

    def test_ocr_request_defaults(self):
        req = OCRRequest(track_config_key="track-xyz789")
        assert req.model_id == "basketball-jersey-numbers-ocr/3"
        assert req.n_consecutive == 3
        assert req.ocr_interval == 5
        assert req.track_config_key == "track-xyz789"

    def test_render_request_requires_all_config_keys(self):
        with pytest.raises(Exception):
            RenderRequest()  # all 5 config keys are required

    def test_render_request_accepts_all_config_keys(self):
        req = RenderRequest(
            det_config_key="det-001",
            track_config_key="track-002",
            teams_config_key="teams-003",
            jerseys_config_key="jerseys-004",
            court_config_key="court-005",
        )
        assert req.det_config_key == "det-001"
        assert req.track_config_key == "track-002"
        assert req.teams_config_key == "teams-003"
        assert req.jerseys_config_key == "jerseys-004"
        assert req.court_config_key == "court-005"


class TestResponseModels:
    def test_detect_response_skipped_defaults_false(self):
        resp = DetectResponse(
            video_id="vid1",
            config_key="cfg1",
            n_frames=100,
            n_detections=42,
        )
        assert resp.skipped is False

    def test_classify_teams_response_palette_accepts_dict(self):
        palette = {
            "team_a": {"r": 255, "g": 0, "b": 0},
            "team_b": {"r": 0, "g": 0, "b": 255},
        }
        resp = ClassifyTeamsResponse(
            video_id="vid1",
            config_key="cfg1",
            palette=palette,
        )
        assert resp.palette == palette
        assert resp.skipped is False

    def test_pipeline_status_response_with_nested_stage_status(self):
        stages = {
            "detect": StageStatusResponse(
                status="completed",
                config_key="det-001",
                started_at="2026-03-21T10:00:00Z",
                completed_at="2026-03-21T10:05:00Z",
                duration_ms=300000,
                error=None,
            ),
            "track": StageStatusResponse(
                status="pending",
            ),
        }
        resp = PipelineStatusResponse(video_id="vid1", stages=stages)
        assert resp.video_id == "vid1"
        assert resp.stages["detect"].status == "completed"
        assert resp.stages["detect"].config_key == "det-001"
        assert resp.stages["detect"].duration_ms == 300000
        assert resp.stages["track"].status == "pending"
        assert resp.stages["track"].config_key is None
        assert resp.stages["track"].error is None

    def test_stage_status_response_optional_fields_default_none(self):
        resp = StageStatusResponse(status="pending")
        assert resp.config_key is None
        assert resp.started_at is None
        assert resp.completed_at is None
        assert resp.duration_ms is None
        assert resp.error is None

    def test_track_response_skipped_defaults_false(self):
        resp = TrackResponse(
            video_id="vid1",
            config_key="cfg1",
            n_frames=50,
            n_tracks=10,
        )
        assert resp.skipped is False


class TestInferenceSchemas:
    def test_inference_request_defaults_empty_dicts(self):
        req = InferenceRequest(video_id="vid1")
        assert req.params == {}
        assert req.upstream_configs == {}

    def test_inference_request_accepts_populated_dicts(self):
        req = InferenceRequest(
            video_id="vid1",
            params={"confidence": 0.5},
            upstream_configs={"det_config_key": "det-001"},
        )
        assert req.params == {"confidence": 0.5}
        assert req.upstream_configs == {"det_config_key": "det-001"}

    def test_inference_response_error_defaults_none(self):
        resp = InferenceResponse(
            status="ok",
            config_key="cfg1",
            output_path="/data/output.json",
        )
        assert resp.error is None

    def test_inference_response_captures_error(self):
        resp = InferenceResponse(
            status="error",
            config_key="cfg1",
            output_path="",
            error="CUDA out of memory",
        )
        assert resp.error == "CUDA out of memory"
        assert resp.status == "error"
