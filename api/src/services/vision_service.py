"""Async HTTP client for GPU inference services."""

import httpx
from typing import Any


class VisionService:
    """Calls inference-roboflow and inference-vision GPU services over HTTP."""

    def __init__(
        self,
        roboflow_url: str = "http://localhost:8091",
        vision_url: str = "http://localhost:8092",
        timeout: float = 600.0,
    ):
        self.roboflow_url = roboflow_url.rstrip("/")
        self.vision_url = vision_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, base_url: str, path: str, payload: dict) -> dict:
        """POST JSON to a GPU service and return the response dict."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{base_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()

    def _inference_payload(
        self,
        video_id: str,
        params: dict,
        upstream_configs: dict | None = None,
    ) -> dict:
        return {
            "video_id": video_id,
            "params": params,
            "upstream_configs": upstream_configs or {},
        }

    async def detect(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post(self.roboflow_url, "/api/detect", payload)

    async def keypoints(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post(self.roboflow_url, "/api/keypoints", payload)

    async def ocr(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post(self.roboflow_url, "/api/ocr", payload)

    async def track(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post(self.vision_url, "/api/track", payload)

    async def classify_teams(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post(self.vision_url, "/api/classify-teams", payload)
