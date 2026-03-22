"""Tests for settings persistence API."""
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


class TestSettingsSchema:
    def test_default_settings(self):
        from api.src.schemas.settings import AnalysisSettings
        s = AnalysisSettings()
        assert s.game_context.teams["0"].name == "Team A"
        assert s.advanced.confidence == 0.4

    def test_custom_settings(self):
        from api.src.schemas.settings import AnalysisSettings
        s = AnalysisSettings(
            game_context={"teams": {"0": {"name": "Lakers", "color": "#552583"}}, "roster": {"23": "James"}},
            advanced={"confidence": 0.3, "iou_threshold": 0.8, "ocr_interval": 10, "crop_scale": 0.5, "stride": 15},
        )
        assert s.game_context.teams["0"].name == "Lakers"
        assert s.advanced.confidence == 0.3


class TestSettingsEndpoints:
    def test_get_default_settings(self, client, tmp_path, monkeypatch):
        from api.src import config as cfg_mod
        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)
        resp = client.get("/api/settings/LPDnemFoqVk")
        assert resp.status_code == 200
        body = resp.json()
        assert body["game_context"]["teams"]["0"]["name"] == "Team A"
        assert body["advanced"]["confidence"] == 0.4

    def test_put_and_get_settings(self, client, tmp_path, monkeypatch):
        from api.src import config as cfg_mod
        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)
        settings_data = {
            "game_context": {"teams": {"0": {"name": "Knicks", "color": "#006BB6"}}, "roster": {"11": "Brunson"}},
            "advanced": {"confidence": 0.3, "iou_threshold": 0.9, "ocr_interval": 5, "crop_scale": 0.4, "stride": 30},
        }
        resp = client.put("/api/settings/LPDnemFoqVk", json=settings_data)
        assert resp.status_code == 200
        resp = client.get("/api/settings/LPDnemFoqVk")
        assert resp.status_code == 200
        assert resp.json()["game_context"]["teams"]["0"]["name"] == "Knicks"
        assert resp.json()["game_context"]["roster"]["11"] == "Brunson"

    def test_settings_endpoint_registered(self, app):
        paths = list(app.openapi()["paths"].keys())
        assert any("/api/settings" in p for p in paths)
