"""Async HTTP client for the GPU inference service."""

import httpx
from typing import Any

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore


class VisionService:
    """Calls the basket-tube GPU inference service over HTTP."""

    def __init__(
        self,
        gpu_url: str = "http://localhost:8090",
        timeout: float = 3600.0,
    ):
        self.gpu_url = gpu_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to the GPU service and return the response dict."""
        stage = path.rsplit("/", 1)[-1]  # e.g. "detect" from "/api/detect"
        video_id = payload.get("video_id", "unknown")

        if logfire:
            with logfire.span("gpu.{stage}", stage=stage, video_id=video_id):
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(f"{self.gpu_url}{path}", json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                    logfire.info("gpu.{stage} complete", stage=stage, video_id=video_id,
                                config_key=result.get("config_key", ""),
                                status=result.get("status", ""))
                    return result
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.gpu_url}{path}", json=payload)
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
        return await self._post("/api/detect", payload)

    async def keypoints(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/keypoints", payload)

    async def ocr(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/ocr", payload)

    async def track(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/track", payload)

    async def classify_teams(self, video_id: str, params: dict, **kw: Any) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/classify-teams", payload)
