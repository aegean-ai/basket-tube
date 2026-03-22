"""Tests for the vision service HTTP client (mocked HTTP)."""

import pytest
from unittest.mock import AsyncMock, patch
from api.src.services.vision_service import VisionService


@pytest.fixture
def service():
    return VisionService(
        roboflow_url="http://localhost:8091",
        vision_url="http://localhost:8092",
        timeout=10.0,
    )


class TestVisionService:
    @pytest.mark.asyncio
    async def test_call_detect(self, service):
        mock_resp = {"status": "ok", "config_key": "c-abc1234", "output_path": "analysis/detections/c-abc1234/test.json", "error": None}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.detect("test-video-id", {"confidence": 0.4})
            assert result["status"] == "ok"
            assert result["config_key"] == "c-abc1234"

    @pytest.mark.asyncio
    async def test_call_track(self, service):
        mock_resp = {"status": "ok", "config_key": "c-def5678", "output_path": "analysis/tracks/c-def5678/test.json", "error": None}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.track("test-video-id", {}, upstream_configs={"detections": "c-abc1234"})
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_call_classify_teams(self, service):
        mock_resp = {"status": "ok", "config_key": "c-111", "output_path": "x", "error": None}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.classify_teams("vid", {"stride": 30})
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_call_keypoints(self, service):
        mock_resp = {"status": "ok", "config_key": "c-222", "output_path": "x", "error": None}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.keypoints("vid", {})
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_call_ocr(self, service):
        mock_resp = {"status": "ok", "config_key": "c-333", "output_path": "x", "error": None}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.ocr("vid", {})
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_gpu_error_returned(self, service):
        mock_resp = {"status": "error", "config_key": "", "output_path": "", "error": "OOM"}
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_resp):
            result = await service.detect("vid", {})
            assert result["status"] == "error"
            assert result["error"] == "OOM"

    def test_payload_structure(self, service):
        payload = service._inference_payload("vid123", {"confidence": 0.4}, {"detections": "c-abc"})
        assert payload == {"video_id": "vid123", "params": {"confidence": 0.4}, "upstream_configs": {"detections": "c-abc"}}

    def test_payload_defaults_empty_upstream(self, service):
        payload = service._inference_payload("vid123", {})
        assert payload["upstream_configs"] == {}
