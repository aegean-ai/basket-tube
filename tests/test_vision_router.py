"""Tests for the vision pipeline router."""
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    from api.src.main import create_app
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestVisionRouterRegistered:
    def test_vision_endpoints_exist(self, app):
        paths = list(app.openapi()["paths"].keys())
        assert any("/api/vision/detect" in p for p in paths)
        assert any("/api/vision/track" in p for p in paths)
        assert any("/api/vision/classify-teams" in p for p in paths)
        assert any("/api/vision/ocr" in p for p in paths)
        assert any("/api/vision/court-map" in p for p in paths)
        assert any("/api/vision/render" in p for p in paths)
        assert any("/api/vision/status" in p for p in paths)


class TestStatusEndpoint:
    def test_status_unknown_video(self, client):
        resp = client.get("/api/vision/status/nonexistent")
        assert resp.status_code == 404


class TestDetectEndpoint:
    def test_detect_unknown_video_returns_404(self, client):
        resp = client.post("/api/vision/detect/nonexistent")
        assert resp.status_code == 404


class TestDependencyEnforcement:
    def test_track_without_detections_returns_409(self, client):
        resp = client.post(
            "/api/vision/track/LPDnemFoqVk",
            json={"det_config_key": "c-nonexistent"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert "missing" in body["detail"]

    def test_ocr_without_tracks_returns_409(self, client):
        resp = client.post(
            "/api/vision/ocr/LPDnemFoqVk",
            json={"track_config_key": "c-nonexistent"},
        )
        assert resp.status_code == 409


class TestSkipOnExists:
    def test_detect_skips_when_output_exists(self, client, tmp_path, monkeypatch):
        from api.src import config as cfg_mod
        from api.src.artifacts import artifact_path, config_key

        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)

        params = {"model_id": "basketball-player-detection-3-ycjdo/4", "confidence": 0.4, "iou_threshold": 0.9}
        cfg_key = config_key(params)
        stem = "Warriors & Lakers Instant Classic - 2021 Play-In Tournament"
        out = artifact_path(tmp_path, "detections", cfg_key, stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_frames": 10, "n_detections": 50}))

        resp = client.post("/api/vision/detect/LPDnemFoqVk")
        assert resp.status_code == 200
        assert resp.json()["skipped"] is True
