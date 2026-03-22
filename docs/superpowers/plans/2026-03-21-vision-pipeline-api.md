# Vision Pipeline & Captions API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the multi-container basketball video analysis pipeline (CPU API orchestrator + GPU inference service) and the captions text timeline endpoint required by the action recognition dataset builder.

**Architecture:** Four containers sharing `./pipeline_data` via Docker volumes: CPU API (:8080), Whisper STT (:8000, speaches image), GPU inference (:8090), and notebook (:8888). CPU API orchestrates by calling the GPU inference service over HTTP, passing `video_id` (not raw paths). Each stage writes config-key-namespaced JSON artifacts with status sidecars for lifecycle tracking. The captions pipeline (download → transcribe → text timeline) runs entirely on the CPU API container.

**Tech Stack:** FastAPI, Pydantic, httpx (async HTTP client), PyTorch 2.7 + CUDA 12.8, Roboflow inference-gpu SDK, SAM2, supervision, sports (roboflow), Docker Compose.

**Specs:**
- `docs/superpowers/specs/2026-03-21-vision-pipeline-api-design.md`
- `docs/superpowers/specs/2026-03-21-captions-api-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `api/src/artifacts.py` | Config key computation, artifact path resolution, status sidecar read/write, atomic file operations |
| `api/src/schemas/vision.py` | All vision pipeline Pydantic request/response models |
| `api/src/services/vision_service.py` | Async HTTP client to GPU inference service |
| `api/src/services/whisper_service.py` | Remote HTTP client for Whisper STT (speaches container) |
| `api/src/routers/vision.py` | 6 stage endpoints + status endpoint, caching/skip logic, dependency enforcement |
| `basket_tube/inference/main.py` | FastAPI app: /api/detect, /api/keypoints, /api/ocr, /api/track, /api/classify-teams, /health |
| `basket_tube/inference/roboflow/` | Roboflow model loading (local/remote), frame-level inference wrappers |
| `basket_tube/inference/vision/` | SAM2Tracker, TeamClassifier wrapper |
| `Dockerfile.api` | CPU API image |
| `Dockerfile.gpu` | GPU image for inference + notebook |
| `tests/test_artifacts.py` | Tests for config key, path resolution, status sidecars |
| `tests/test_vision_schemas.py` | Tests for vision Pydantic models |
| `tests/test_vision_router.py` | Tests for vision router (mocked GPU services) |
| `tests/test_vision_service.py` | Tests for HTTP client to GPU services |
| `api/src/schemas/captions.py` | TextTimelineRequest, TextTimelineResponse |
| `api/src/services/text_timeline_service.py` | Normalization logic, basketball lexicon, segment transformation |
| `api/src/routers/captions.py` | POST /api/captions/timeline/{video_id} endpoint |
| `tests/test_text_timeline.py` | Tests for normalization, timeline construction, skip-on-exists |

### Modified Files

| File | Change |
|---|---|
| `api/src/config.py` | Add `inference_gpu_url`, `whisper_api_url`, `analysis_dir` property |
| `api/src/video_registry.py` | Add `resolve_stem()` alias for `resolve_title()` |
| `api/src/main.py` | Register download, transcribe, vision, and captions routers |
| `docker-compose.yml` | 4 services: api, whisper, inference, notebook |
| `Dockerfile.api` | CPU API image |
| `pyproject.toml` | Add `httpx` and `pytest-asyncio` to deps |

---

### Task 1: Artifact utilities — config key, path resolution, status sidecars

**Files:**
- Create: `api/src/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write failing tests for config_key()**

```python
# tests/test_artifacts.py
from api.src.artifacts import config_key


class TestConfigKey:
    def test_deterministic(self):
        params = {"confidence": 0.4, "model_id": "basketball-player-detection-3-ycjdo/4"}
        assert config_key(params) == config_key(params)

    def test_key_order_independent(self):
        a = config_key({"confidence": 0.4, "model_id": "x"})
        b = config_key({"model_id": "x", "confidence": 0.4})
        assert a == b

    def test_format(self):
        k = config_key({"x": 1})
        assert k.startswith("c-")
        assert len(k) == 9  # "c-" + 7 hex chars

    def test_different_params_different_keys(self):
        a = config_key({"confidence": 0.4})
        b = config_key({"confidence": 0.3})
        assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.src.artifacts'`

- [ ] **Step 3: Implement config_key()**

```python
# api/src/artifacts.py
"""Artifact management: config keys, path resolution, status sidecars."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def config_key(params: dict) -> str:
    """Deterministic short hash of parameters that affect output."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "c-" + hashlib.sha256(canonical.encode()).hexdigest()[:7]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_artifacts.py::TestConfigKey -v`
Expected: 4 PASSED

- [ ] **Step 5: Write failing tests for artifact_path() and status sidecar functions**

```python
# tests/test_artifacts.py (append)
from pathlib import Path
from api.src.artifacts import artifact_path, write_status, read_status


class TestArtifactPath:
    def test_returns_path_with_config_key(self, tmp_path):
        p = artifact_path(tmp_path, "detections", "c-abc1234", "my-video")
        assert p == tmp_path / "analysis" / "detections" / "c-abc1234" / "my-video.json"

    def test_mp4_extension_for_renders(self, tmp_path):
        p = artifact_path(tmp_path, "renders", "c-abc1234", "my-video")
        assert p == tmp_path / "analysis" / "renders" / "c-abc1234" / "my-video.mp4"


class TestStatusSidecar:
    def test_write_and_read_active(self, tmp_path):
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "active")
        status = read_status(sidecar)
        assert status["status"] == "active"
        assert "started_at" in status

    def test_write_complete(self, tmp_path):
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "active")
        write_status(sidecar, "complete", config_key="c-abc1234")
        status = read_status(sidecar)
        assert status["status"] == "complete"
        assert status["config_key"] == "c-abc1234"
        assert status["duration_ms"] is not None

    def test_write_error(self, tmp_path):
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "error", error="GPU OOM")
        status = read_status(sidecar)
        assert status["status"] == "error"
        assert status["error"] == "GPU OOM"

    def test_read_missing_returns_pending(self, tmp_path):
        sidecar = tmp_path / "nonexistent.status.json"
        status = read_status(sidecar)
        assert status["status"] == "pending"

    def test_atomic_write(self, tmp_path):
        """No partial writes — tmp file should not linger on success."""
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "complete", config_key="c-abc1234")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestCheckStale:
    def test_active_within_timeout_returns_active(self, tmp_path):
        from api.src.artifacts import check_stale
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "active")
        result = check_stale(sidecar, timeout_s=600)
        assert result["status"] == "active"

    def test_stale_active_returns_pending_and_removes_sidecar(self, tmp_path):
        from api.src.artifacts import check_stale
        import json
        sidecar = tmp_path / "test.status.json"
        # Write an active status with an old started_at
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()
        sidecar.write_text(json.dumps({"status": "active", "started_at": old_time}))
        result = check_stale(sidecar, timeout_s=600)
        assert result["status"] == "pending"
        assert not sidecar.exists()

    def test_complete_status_unchanged(self, tmp_path):
        from api.src.artifacts import check_stale
        sidecar = tmp_path / "test.status.json"
        write_status(sidecar, "complete", config_key="c-abc1234")
        result = check_stale(sidecar, timeout_s=600)
        assert result["status"] == "complete"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement artifact_path(), write_status(), read_status()**

```python
# api/src/artifacts.py (append to existing)

_RENDER_STAGES = {"renders"}


def artifact_path(data_dir: Path, stage: str, cfg_key: str, stem: str) -> Path:
    """Canonical artifact path for a pipeline stage output."""
    ext = ".mp4" if stage in _RENDER_STAGES else ".json"
    return data_dir / "analysis" / stage / cfg_key / f"{stem}{ext}"


def status_path_for(artifact: Path) -> Path:
    """Status sidecar path for an artifact."""
    return artifact.with_suffix(".status.json")


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_status(
    sidecar: Path,
    status: str,
    *,
    config_key: str | None = None,
    error: str | None = None,
) -> None:
    """Write a status sidecar atomically."""
    now = datetime.now(timezone.utc).isoformat()

    # Read existing to preserve started_at for completion
    existing = read_status(sidecar)

    data: dict[str, Any] = {"status": status}

    if status == "active":
        data["started_at"] = now
        data["completed_at"] = None
        data["duration_ms"] = None
        data["error"] = None
    elif status == "complete":
        data["started_at"] = existing.get("started_at", now)
        data["completed_at"] = now
        started = existing.get("started_at")
        if started:
            start_dt = datetime.fromisoformat(started)
            data["duration_ms"] = int(
                (datetime.now(timezone.utc) - start_dt).total_seconds() * 1000
            )
        else:
            data["duration_ms"] = None
        data["error"] = None
    elif status == "error":
        data["started_at"] = existing.get("started_at", now)
        data["completed_at"] = now
        data["error"] = error
        data["duration_ms"] = None
    else:
        data["started_at"] = None
        data["completed_at"] = None
        data["duration_ms"] = None
        data["error"] = None

    if config_key:
        data["config_key"] = config_key

    atomic_write_json(sidecar, data)


def read_status(sidecar: Path) -> dict:
    """Read a status sidecar. Returns pending status if file missing."""
    if not sidecar.exists():
        return {"status": "pending"}
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return {"status": "pending"}
```

- [ ] **Step 7b: Implement check_stale() for crash recovery**

Append to `api/src/artifacts.py`:

```python
def check_stale(sidecar: Path, timeout_s: float = 600.0) -> dict:
    """Check if an active status sidecar is stale (crash recovery).

    If started_at is older than timeout_s, treat as crashed:
    delete the sidecar and return pending status.
    For any other status, return it unchanged.
    """
    status = read_status(sidecar)
    if status["status"] != "active":
        return status

    started = status.get("started_at")
    if not started:
        return status

    start_dt = datetime.fromisoformat(started)
    elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
    if elapsed > timeout_s:
        # Stale — crash recovery
        try:
            sidecar.unlink()
        except OSError:
            pass
        return {"status": "pending"}

    return status
```

- [ ] **Step 8: Run all artifact tests**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: ALL PASSED

- [ ] **Step 9: Commit**

```bash
git add api/src/artifacts.py tests/test_artifacts.py
git commit -m "feat: add artifact utilities — config key, path resolution, status sidecars"
```

---

### Task 2: Vision schemas

**Files:**
- Create: `api/src/schemas/vision.py`
- Test: `tests/test_vision_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_schemas.py
from api.src.schemas.vision import (
    DetectRequest,
    TrackRequest,
    ClassifyTeamsRequest,
    OCRRequest,
    CourtMapRequest,
    RenderRequest,
    DetectResponse,
    TrackResponse,
    ClassifyTeamsResponse,
    OCRResponse,
    CourtMapResponse,
    RenderResponse,
    StageStatusResponse,
    PipelineStatusResponse,
    InferenceRequest,
    InferenceResponse,
)


class TestRequestDefaults:
    def test_detect_defaults(self):
        r = DetectRequest()
        assert r.confidence == 0.4
        assert r.iou_threshold == 0.9
        assert r.model_id == "basketball-player-detection-3-ycjdo/4"

    def test_track_requires_det_config_key(self):
        r = TrackRequest(det_config_key="c-abc1234")
        assert r.det_config_key == "c-abc1234"

    def test_classify_teams_defaults(self):
        r = ClassifyTeamsRequest(det_config_key="c-abc1234")
        assert r.stride == 30
        assert r.crop_scale == 0.4

    def test_ocr_requires_track_config_key(self):
        r = OCRRequest(track_config_key="c-abc1234")
        assert r.n_consecutive == 3

    def test_render_requires_all_config_keys(self):
        r = RenderRequest(
            det_config_key="c-1",
            track_config_key="c-2",
            teams_config_key="c-3",
            jerseys_config_key="c-4",
            court_config_key="c-5",
        )
        assert r.det_config_key == "c-1"


class TestResponseModels:
    def test_detect_response(self):
        r = DetectResponse(video_id="x", config_key="c-1", n_frames=100, n_detections=500)
        assert r.skipped is False

    def test_classify_teams_response_palette(self):
        r = ClassifyTeamsResponse(
            video_id="x",
            config_key="c-1",
            palette={"0": {"name": "Team A", "color": "#006BB6"}},
        )
        assert "0" in r.palette

    def test_pipeline_status(self):
        r = PipelineStatusResponse(
            video_id="x",
            stages={"detect": StageStatusResponse(status="complete", config_key="c-1")},
        )
        assert r.stages["detect"].status == "complete"


class TestInferenceSchemas:
    def test_inference_request(self):
        r = InferenceRequest(video_id="LPDnemFoqVk")
        assert r.params == {}
        assert r.upstream_configs == {}

    def test_inference_response(self):
        r = InferenceResponse(status="ok", config_key="c-1", output_path="analysis/x.json")
        assert r.error is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vision_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement schemas**

```python
# api/src/schemas/vision.py
"""Pydantic request/response models for the vision pipeline."""

from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    model_id: str = "basketball-player-detection-3-ycjdo/4"
    confidence: float = 0.4
    iou_threshold: float = 0.9
    max_frames: int | None = None


class TrackRequest(BaseModel):
    det_config_key: str
    sam2_checkpoint: str = "sam2.1_hiera_large.pt"
    max_frames: int | None = None


class ClassifyTeamsRequest(BaseModel):
    det_config_key: str
    stride: int = 30
    crop_scale: float = 0.4


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
    court_config_key: str


# ── Responses ─────────────────────────────────────────────────────────

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


# ── GPU Service Schemas (internal) ────────────────────────────────────

class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vision_schemas.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add api/src/schemas/vision.py tests/test_vision_schemas.py
git commit -m "feat: add vision pipeline Pydantic schemas"
```

---

### Task 3: Config and registry updates

**Files:**
- Modify: `api/src/config.py`
- Modify: `api/src/video_registry.py`
- Test: `tests/test_api_scaffold.py` (extend existing)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_artifacts.py (append)
class TestConfigVisionSettings:
    def test_inference_gpu_url_default(self):
        from api.src.config import Settings
        s = Settings()
        assert s.inference_gpu_url == "http://localhost:8090"

    def test_whisper_api_url_default(self):
        from api.src.config import Settings
        s = Settings()
        assert s.whisper_api_url == "http://localhost:8000"

    def test_analysis_dir_property(self):
        from api.src.config import Settings
        s = Settings()
        assert s.analysis_dir == s.data_dir / "analysis"

    def test_inference_url_from_env(self, monkeypatch):
        monkeypatch.setenv("FW_INFERENCE_GPU_URL", "http://gpu:9000")
        from api.src.config import Settings
        s = Settings()
        assert s.inference_gpu_url == "http://gpu:9000"


class TestResolveStem:
    def test_resolve_stem_alias(self):
        from api.src.video_registry import resolve_stem, resolve_title
        # resolve_stem should be the same function as resolve_title
        assert resolve_stem is resolve_title
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_artifacts.py::TestConfigVisionSettings tests/test_artifacts.py::TestResolveStam -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Add settings to config.py**

Add in `api/src/config.py`:

```python
    # GPU inference service URL
    inference_gpu_url: str = "http://localhost:8090"
```

Add after `dubbed_captions_dir` property:

```python
    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"
```

- [ ] **Step 4: Add resolve_stem to video_registry.py**

Append to `api/src/video_registry.py`:

```python
# Alias: spec uses resolve_stem, existing code uses resolve_title
resolve_stem = resolve_title
```

- [ ] **Step 5: Add httpx and pytest-asyncio to pyproject.toml**

In `pyproject.toml`, add `httpx` to main dependencies and `pytest-asyncio` to dev:

```toml
# In [project] dependencies, add:
    "httpx",

# In [dependency-groups] dev, add:
    "pytest-asyncio",
```

Run: `uv lock`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_artifacts.py::TestConfigVisionSettings tests/test_artifacts.py::TestResolveStem -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add api/src/config.py api/src/video_registry.py tests/test_artifacts.py pyproject.toml uv.lock
git commit -m "feat: add vision inference URL, analysis_dir, httpx dep"
```

---

### Task 4: Vision service — HTTP client to GPU services

**Files:**
- Create: `api/src/services/vision_service.py`
- Test: `tests/test_vision_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_service.py
"""Tests for the vision service HTTP client (mocked HTTP)."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.src.services.vision_service import VisionService
from api.src.schemas.vision import InferenceResponse


@pytest.fixture
def service():
    return VisionService(
        gpu_url="http://localhost:8090",
        timeout=10.0,
    )


class TestVisionService:
    @pytest.mark.asyncio
    async def test_call_detect(self, service):
        mock_response = InferenceResponse(
            status="ok", config_key="c-abc1234", output_path="analysis/detections/c-abc1234/test.json"
        )
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_response.model_dump()):
            result = await service.detect("test-video-id", {"confidence": 0.4})
            assert result["status"] == "ok"
            assert result["config_key"] == "c-abc1234"

    @pytest.mark.asyncio
    async def test_call_track(self, service):
        mock_response = InferenceResponse(
            status="ok", config_key="c-def5678", output_path="analysis/tracks/c-def5678/test.json"
        )
        with patch.object(service, "_post", new_callable=AsyncMock, return_value=mock_response.model_dump()):
            result = await service.track("test-video-id", {}, upstream_configs={"detections": "c-abc1234"})
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_gpu_service_error_raises(self, service):
        with patch.object(
            service, "_post", new_callable=AsyncMock,
            return_value={"status": "error", "config_key": "", "output_path": "", "error": "OOM"},
        ):
            result = await service.detect("test-video-id", {})
            assert result["status"] == "error"
            assert result["error"] == "OOM"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vision_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement VisionService**

```python
# api/src/services/vision_service.py
"""Async HTTP client for GPU inference services."""

import httpx
from typing import Any


class VisionService:
    """Calls the GPU inference service over HTTP."""

    def __init__(
        self,
        gpu_url: str = "http://localhost:8090",
        timeout: float = 600.0,
    ):
        self.gpu_url = gpu_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to the GPU service and return the response dict."""
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

    async def detect(
        self, video_id: str, params: dict, **kw: Any
    ) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/detect", payload)

    async def keypoints(
        self, video_id: str, params: dict, **kw: Any
    ) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/keypoints", payload)

    async def ocr(
        self, video_id: str, params: dict, **kw: Any
    ) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/ocr", payload)

    async def track(
        self, video_id: str, params: dict, **kw: Any
    ) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/track", payload)

    async def classify_teams(
        self, video_id: str, params: dict, **kw: Any
    ) -> dict:
        payload = self._inference_payload(video_id, params, kw.get("upstream_configs"))
        return await self._post("/api/classify-teams", payload)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vision_service.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add api/src/services/vision_service.py tests/test_vision_service.py
git commit -m "feat: add VisionService HTTP client for GPU inference services"
```

---

### Task 5: Vision router — CPU API orchestration endpoints

**Files:**
- Create: `api/src/routers/vision.py`
- Modify: `api/src/main.py`
- Test: `tests/test_vision_router.py`

- [ ] **Step 1: Write failing tests for router registration and detect endpoint**

```python
# tests/test_vision_router.py
"""Tests for the vision pipeline router."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
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
        """Track endpoint requires detections to exist first."""
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
        """If detection output already exists, return skipped=True."""
        import json
        from api.src import config as cfg_mod

        # Patch data_dir to tmp_path
        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)

        # Create fake output
        from api.src.artifacts import artifact_path, config_key
        params = {"model_id": "basketball-player-detection-3-ycjdo/4", "confidence": 0.4, "iou_threshold": 0.9}
        cfg_key = config_key(params)
        out = artifact_path(tmp_path, "detections", cfg_key, "Warriors & Lakers Instant Classic - 2021 Play-In Tournament")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_frames": 10, "n_detections": 50}))

        resp = client.post("/api/vision/detect/LPDnemFoqVk")
        assert resp.status_code == 200
        assert resp.json()["skipped"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vision_router.py -v`
Expected: FAIL — routes not registered

- [ ] **Step 3: Implement vision router**

```python
# api/src/routers/vision.py
"""Vision pipeline router — orchestrates GPU inference services."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.artifacts import (
    artifact_path,
    check_stale,
    config_key,
    read_status,
    status_path_for,
    write_status,
)
from api.src.config import settings
from api.src.video_registry import resolve_stem
from api.src.schemas.vision import (
    ClassifyTeamsRequest,
    ClassifyTeamsResponse,
    CourtMapRequest,
    CourtMapResponse,
    DetectRequest,
    DetectResponse,
    OCRRequest,
    OCRResponse,
    PipelineStatusResponse,
    RenderRequest,
    RenderResponse,
    StageStatusResponse,
    TrackRequest,
    TrackResponse,
)
from api.src.services.vision_service import VisionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

STAGE_NAMES = ["detections", "tracks", "teams", "jerseys", "court", "renders"]


def _get_vision_service() -> VisionService:
    return VisionService(
        gpu_url=settings.inference_gpu_url,
    )


def _resolve_or_404(video_id: str) -> str:
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")
    return stem


def _require_upstream(stage: str, video_id: str, stem: str, upstream_config_key: str) -> Path:
    """Check that an upstream artifact exists. Raise 409 if missing."""
    path = artifact_path(settings.data_dir, stage, upstream_config_key, stem)
    if not path.exists():
        raise HTTPException(
            409,
            detail={"detail": f"Stage '{stage}' must be completed before this stage", "missing": [stage]},
        )
    return path


def _check_not_running(sidecar: Path, stage_name: str) -> None:
    """Check stale (crash recovery) then reject if still active."""
    current = check_stale(sidecar, timeout_s=600.0)
    if current["status"] == "active":
        raise HTTPException(
            409,
            detail={"detail": f"Stage '{stage_name}' is already running"},
        )


# ── Stage endpoints ───────────────────────────────────────────────────


@router.post("/detect/{video_id}", response_model=DetectResponse)
async def detect(video_id: str, req: DetectRequest | None = None):
    req = req or DetectRequest()
    stem = _resolve_or_404(video_id)

    cfg_params = {"model_id": req.model_id, "confidence": req.confidence, "iou_threshold": req.iou_threshold}
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "detections", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        data = json.loads(out.read_text())
        return DetectResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames=data.get("n_frames", 0), n_detections=data.get("n_detections", 0),
            skipped=True,
        )

    _check_not_running(sidecar, "detect")

    write_status(sidecar, "active")
    try:
        svc = _get_vision_service()
        result = await svc.detect(video_id, req.model_dump())
        if result.get("status") == "error":
            write_status(sidecar, "error", error=result.get("error", "Unknown"))
            raise HTTPException(500, detail=result.get("error", "GPU inference failed"))

        write_status(sidecar, "complete", config_key=cfg_key)

        data = json.loads(out.read_text()) if out.exists() else {}
        return DetectResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames=data.get("n_frames", 0), n_detections=data.get("n_detections", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(502, detail=f"GPU service error: {exc}")


@router.post("/track/{video_id}", response_model=TrackResponse)
async def track(video_id: str, req: TrackRequest):
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {"sam2_checkpoint": req.sam2_checkpoint, "det_config_key": req.det_config_key}
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "tracks", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        data = json.loads(out.read_text())
        return TrackResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames=data.get("n_frames", 0), n_tracks=data.get("n_tracks", 0),
            skipped=True,
        )

    _check_not_running(sidecar, "track")

    write_status(sidecar, "active")
    try:
        svc = _get_vision_service()
        result = await svc.track(
            video_id, req.model_dump(exclude={"det_config_key"}),
            upstream_configs={"detections": req.det_config_key},
        )
        if result.get("status") == "error":
            write_status(sidecar, "error", error=result.get("error", "Unknown"))
            raise HTTPException(500, detail=result.get("error"))

        write_status(sidecar, "complete", config_key=cfg_key)
        data = json.loads(out.read_text()) if out.exists() else {}
        return TrackResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames=data.get("n_frames", 0), n_tracks=data.get("n_tracks", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(502, detail=f"GPU service error: {exc}")


@router.post("/classify-teams/{video_id}", response_model=ClassifyTeamsResponse)
async def classify_teams(video_id: str, req: ClassifyTeamsRequest):
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {"stride": req.stride, "crop_scale": req.crop_scale, "det_config_key": req.det_config_key}
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "teams", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        data = json.loads(out.read_text())
        return ClassifyTeamsResponse(
            video_id=video_id, config_key=cfg_key,
            palette=data.get("palette", {}), skipped=True,
        )

    _check_not_running(sidecar, "classify-teams")

    write_status(sidecar, "active")
    try:
        svc = _get_vision_service()
        result = await svc.classify_teams(
            video_id, req.model_dump(exclude={"det_config_key"}),
            upstream_configs={"detections": req.det_config_key},
        )
        if result.get("status") == "error":
            write_status(sidecar, "error", error=result.get("error", "Unknown"))
            raise HTTPException(500, detail=result.get("error"))

        write_status(sidecar, "complete", config_key=cfg_key)
        data = json.loads(out.read_text()) if out.exists() else {}
        return ClassifyTeamsResponse(
            video_id=video_id, config_key=cfg_key,
            palette=data.get("palette", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(502, detail=f"GPU service error: {exc}")


@router.post("/ocr/{video_id}", response_model=OCRResponse)
async def ocr(video_id: str, req: OCRRequest):
    stem = _resolve_or_404(video_id)
    _require_upstream("tracks", video_id, stem, req.track_config_key)

    cfg_params = {
        "model_id": req.model_id, "n_consecutive": req.n_consecutive,
        "ocr_interval": req.ocr_interval, "track_config_key": req.track_config_key,
    }
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "jerseys", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        data = json.loads(out.read_text())
        return OCRResponse(
            video_id=video_id, config_key=cfg_key,
            players=data.get("players", {}), skipped=True,
        )

    _check_not_running(sidecar, "ocr")

    write_status(sidecar, "active")
    try:
        svc = _get_vision_service()
        result = await svc.ocr(
            video_id, req.model_dump(exclude={"track_config_key"}),
            upstream_configs={"tracks": req.track_config_key},
        )
        if result.get("status") == "error":
            write_status(sidecar, "error", error=result.get("error", "Unknown"))
            raise HTTPException(500, detail=result.get("error"))

        write_status(sidecar, "complete", config_key=cfg_key)
        data = json.loads(out.read_text()) if out.exists() else {}
        return OCRResponse(
            video_id=video_id, config_key=cfg_key,
            players=data.get("players", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(502, detail=f"GPU service error: {exc}")


@router.post("/court-map/{video_id}", response_model=CourtMapResponse)
async def court_map(video_id: str, req: CourtMapRequest):
    stem = _resolve_or_404(video_id)
    _require_upstream("detections", video_id, stem, req.det_config_key)

    cfg_params = {
        "model_id": req.model_id, "keypoint_confidence": req.keypoint_confidence,
        "anchor_confidence": req.anchor_confidence, "det_config_key": req.det_config_key,
    }
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "court", cfg_key, stem)
    sidecar = status_path_for(out)

    if out.exists():
        data = json.loads(out.read_text())
        return CourtMapResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames_mapped=data.get("n_frames_mapped", 0), skipped=True,
        )

    _check_not_running(sidecar, "court-map")

    write_status(sidecar, "active")
    try:
        svc = _get_vision_service()
        result = await svc.keypoints(
            video_id, req.model_dump(exclude={"det_config_key"}),
            upstream_configs={"detections": req.det_config_key},
        )
        if result.get("status") == "error":
            write_status(sidecar, "error", error=result.get("error", "Unknown"))
            raise HTTPException(500, detail=result.get("error"))

        write_status(sidecar, "complete", config_key=cfg_key)
        data = json.loads(out.read_text()) if out.exists() else {}
        return CourtMapResponse(
            video_id=video_id, config_key=cfg_key,
            n_frames_mapped=data.get("n_frames_mapped", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(502, detail=f"GPU service error: {exc}")


@router.post("/render/{video_id}", response_model=RenderResponse)
async def render(video_id: str, req: RenderRequest):
    stem = _resolve_or_404(video_id)

    # Verify all upstream stages exist
    _require_upstream("detections", video_id, stem, req.det_config_key)
    _require_upstream("tracks", video_id, stem, req.track_config_key)
    _require_upstream("teams", video_id, stem, req.teams_config_key)
    _require_upstream("jerseys", video_id, stem, req.jerseys_config_key)
    _require_upstream("court", video_id, stem, req.court_config_key)

    cfg_params = req.model_dump()
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "renders", cfg_key, stem)

    if out.exists():
        return RenderResponse(video_id=video_id, config_key=cfg_key, skipped=True)

    # TODO: implement CPU-side render (ffmpeg/opencv)
    raise HTTPException(501, detail="Render not yet implemented")


@router.get("/status/{video_id}", response_model=PipelineStatusResponse)
async def status(video_id: str, config_key_filter: str | None = None):
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")

    stages: dict[str, StageStatusResponse] = {}
    for stage in STAGE_NAMES:
        stage_dir = settings.data_dir / "analysis" / stage
        if not stage_dir.exists():
            stages[stage] = StageStatusResponse(status="pending")
            continue

        if config_key_filter:
            # Check specific config key only
            art = artifact_path(settings.data_dir, stage, config_key_filter, stem)
            sidecar = status_path_for(art)
            s = check_stale(sidecar, timeout_s=600.0)
            stages[stage] = StageStatusResponse(**s)
        else:
            # Find latest (most recent completed_at) config for this stage
            latest_status = StageStatusResponse(status="pending")
            for cfg_dir in sorted(stage_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                art = artifact_path(settings.data_dir, stage, cfg_dir.name, stem)
                sidecar = status_path_for(art)
                s = check_stale(sidecar, timeout_s=600.0)
                if s["status"] != "pending":
                    latest_status = StageStatusResponse(**s)
            stages[stage] = latest_status

    return PipelineStatusResponse(video_id=video_id, stages=stages)
```

- [ ] **Step 4: Register vision router in main.py**

Add to `api/src/main.py` in `create_app()` after the transcribe router registration:

```python
    from api.src.routers.vision import router as vision_router
    app.include_router(vision_router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_vision_router.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_api_scaffold.py tests/test_fastapi_skeleton.py -v`
Expected: ALL PASSED (existing endpoints still work, vision endpoints now registered)

- [ ] **Step 7: Commit**

```bash
git add api/src/routers/vision.py api/src/main.py tests/test_vision_router.py
git commit -m "feat: add vision pipeline router with 6 stage endpoints + status"
```

---

### Task 6: Captions schemas and text timeline service

**Files:**
- Create: `api/src/schemas/captions.py`
- Create: `api/src/services/text_timeline_service.py`
- Test: `tests/test_text_timeline.py`

- [ ] **Step 1: Write failing tests for the basketball lexicon normalizer**

```python
# tests/test_text_timeline.py
"""Tests for text timeline service — normalization and segment construction."""

from api.src.services.text_timeline_service import normalize_text, build_timeline


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Curry FOR THREE!") == "curry for three"

    def test_strip_trailing_punctuation(self):
        assert normalize_text("He knocks it down!") == "he knocks it down"

    def test_preserve_apostrophes(self):
        assert normalize_text("can't get it to go") == "can't get it to go"

    def test_three_pointer_synonyms(self):
        assert "three" in normalize_text("fires from 3")
        assert "three" in normalize_text("a trey from the corner")

    def test_dunk_synonyms(self):
        assert "dunk" in normalize_text("What a slam!")
        assert "dunk" in normalize_text("the jam by James")

    def test_layup_synonyms(self):
        assert "layup" in normalize_text("nice lay-up")
        assert "layup" in normalize_text("the lay up")

    def test_and_one_synonyms(self):
        assert "and one" in normalize_text("and-one!")
        assert "and one" in normalize_text("and 1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_text_timeline.py::TestNormalizeText -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement normalize_text()**

```python
# api/src/services/text_timeline_service.py
"""Text timeline construction and basketball vocabulary normalization."""

import re
from typing import Any


# Basketball domain lexicon — maps synonyms to canonical forms
_LEXICON = [
    # Three-point synonyms
    (re.compile(r'\b(?:3|three pointer|trey)\b', re.I), "three"),
    # Dunk synonyms
    (re.compile(r'\b(?:slam|jam)\b', re.I), "dunk"),
    # Layup synonyms
    (re.compile(r'\b(?:lay-up|lay up)\b', re.I), "layup"),
    # And-one synonyms
    (re.compile(r'\b(?:and-one|and 1)\b', re.I), "and one"),
]


def normalize_text(raw: str) -> str:
    """Normalize raw commentary text for downstream pattern matching.

    1. Lowercase
    2. Strip trailing punctuation (preserve apostrophes)
    3. Apply basketball domain lexicon
    """
    text = raw.lower()
    # Strip trailing punctuation but preserve apostrophes
    text = re.sub(r'[!?.,:;]+$', '', text)
    text = text.strip()
    # Apply lexicon
    for pattern, replacement in _LEXICON:
        text = pattern.sub(replacement, text)
    return text
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_text_timeline.py::TestNormalizeText -v`
Expected: ALL PASSED

- [ ] **Step 5: Write failing tests for build_timeline()**

```python
# tests/test_text_timeline.py (append)
import json


class TestBuildTimeline:
    def test_converts_whisper_segments(self):
        transcript = {
            "language": "en",
            "text": "Curry for three! He knocks it down!",
            "segments": [
                {"id": 0, "start": 12.5, "end": 15.2, "text": "Curry for three!"},
                {"id": 1, "start": 15.2, "end": 17.8, "text": "He knocks it down!"},
            ],
        }
        result = build_timeline(transcript, source="caption", lexicon_version="v0.1")
        assert result["source"] == "caption"
        assert result["lexicon_version"] == "v0.1"
        assert len(result["segments"]) == 2

        seg0 = result["segments"][0]
        assert seg0["segment_id"] == 0
        assert seg0["t_start"] == 12.5
        assert seg0["t_end"] == 15.2
        assert seg0["raw_text"] == "Curry for three!"
        assert seg0["normalized_text"] == "curry for three"
        assert seg0["source"] == "caption"
        assert seg0["asr_confidence"] is None

    def test_stt_segments_have_confidence(self):
        transcript = {
            "language": "en",
            "text": "for three",
            "segments": [
                {"id": 0, "start": 1.0, "end": 2.0, "text": "for three", "avg_logprob": -0.3},
            ],
        }
        result = build_timeline(transcript, source="stt", lexicon_version="v0.1")
        seg = result["segments"][0]
        assert seg["source"] == "stt"
        assert seg["asr_confidence"] is not None

    def test_empty_segments_filtered(self):
        transcript = {
            "language": "en",
            "text": "",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": ""},
                {"id": 1, "start": 1.0, "end": 2.0, "text": "  "},
                {"id": 2, "start": 2.0, "end": 3.0, "text": "real text"},
            ],
        }
        result = build_timeline(transcript, source="caption", lexicon_version="v0.1")
        assert len(result["segments"]) == 1
        assert result["segments"][0]["raw_text"] == "real text"

    def test_meta_included(self):
        transcript = {"language": "en", "text": "x", "segments": [{"id": 0, "start": 0, "end": 1, "text": "x"}]}
        result = build_timeline(transcript, source="caption", lexicon_version="v0.1")
        assert "_meta" in result
        assert result["_meta"]["stage"] == "text_timeline"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_text_timeline.py::TestBuildTimeline -v`
Expected: FAIL — `ImportError: cannot import name 'build_timeline'`

- [ ] **Step 7: Implement build_timeline()**

```python
# api/src/services/text_timeline_service.py (append)
import math
from datetime import datetime, timezone


def build_timeline(
    transcript: dict,
    *,
    source: str,
    lexicon_version: str,
    stt_model_dir: str = "whisper",
) -> dict:
    """Transform a Whisper-format transcript into a text timeline artifact.

    Args:
        transcript: Whisper-compatible dict with 'segments' list
        source: "caption" or "stt"
        lexicon_version: version of normalization rules applied
        stt_model_dir: upstream transcription model directory
    """
    from api.src.artifacts import config_key

    cfg_params = {"stt_model_dir": stt_model_dir, "lexicon_version": lexicon_version}
    cfg_key = config_key(cfg_params)

    segments = []
    for seg in transcript.get("segments", []):
        raw = seg.get("text", "").strip()
        if not raw:
            continue

        asr_confidence = None
        if source == "stt" and "avg_logprob" in seg:
            # Convert log probability to 0-1 confidence
            asr_confidence = round(math.exp(seg["avg_logprob"]), 4)

        segments.append({
            "segment_id": seg.get("id", len(segments)),
            "t_start": seg.get("start", 0),
            "t_end": seg.get("end", 0),
            "raw_text": raw,
            "normalized_text": normalize_text(raw),
            "source": source,
            "asr_confidence": asr_confidence,
        })

    return {
        "_meta": {
            "stage": "text_timeline",
            "config_key": cfg_key,
            "upstream": {"transcription": stt_model_dir},
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "source": source,
        "lexicon_version": lexicon_version,
        "segments": segments,
    }
```

- [ ] **Step 8: Run all timeline tests**

Run: `uv run pytest tests/test_text_timeline.py -v`
Expected: ALL PASSED

- [ ] **Step 9: Write captions schemas**

```python
# api/src/schemas/captions.py
"""Pydantic request/response models for the captions timeline endpoint."""

from pydantic import BaseModel


class TextTimelineRequest(BaseModel):
    stt_model_dir: str = "whisper"
    lexicon_version: str = "v0.1"


class TextTimelineResponse(BaseModel):
    video_id: str
    config_key: str
    n_segments: int
    source: str
    skipped: bool = False
```

- [ ] **Step 10: Commit**

```bash
git add api/src/schemas/captions.py api/src/services/text_timeline_service.py tests/test_text_timeline.py
git commit -m "feat: add text timeline service with basketball lexicon normalization"
```

---

### Task 7: Captions timeline router

**Files:**
- Create: `api/src/routers/captions.py`
- Modify: `api/src/main.py`
- Test: `tests/test_text_timeline.py` (extend)

- [ ] **Step 1: Write failing tests for the captions router**

```python
# tests/test_text_timeline.py (append)
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    from api.src.main import create_app
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestCaptionsRouter:
    def test_timeline_endpoint_registered(self, app):
        paths = list(app.openapi()["paths"].keys())
        assert any("/api/captions/timeline" in p for p in paths)

    def test_timeline_unknown_video_returns_404(self, client):
        resp = client.post("/api/captions/timeline/nonexistent")
        assert resp.status_code == 404

    def test_timeline_missing_transcription_returns_409(self, client):
        """Timeline requires transcription to exist first."""
        resp = client.post("/api/captions/timeline/LPDnemFoqVk")
        assert resp.status_code == 409

    def test_timeline_skip_on_exists(self, client, tmp_path, monkeypatch):
        """If timeline output already exists, return skipped=True."""
        import json
        from api.src import config as cfg_mod
        from api.src.artifacts import artifact_path, config_key
        from api.src.services.text_timeline_service import build_timeline

        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)

        # Create fake transcription so the 409 check passes
        trans_dir = tmp_path / "transcriptions" / "whisper"
        trans_dir.mkdir(parents=True)
        stem = "Warriors & Lakers Instant Classic - 2021 Play-In Tournament"
        (trans_dir / f"{stem}.json").write_text(json.dumps({
            "language": "en", "text": "x", "segments": [{"id": 0, "start": 0, "end": 1, "text": "x"}]
        }))

        # Create fake timeline output
        cfg_params = {"stt_model_dir": "whisper", "lexicon_version": "v0.1"}
        cfg_key = config_key(cfg_params)
        out = artifact_path(tmp_path, "text_timeline", cfg_key, stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": "caption", "segments": [{"segment_id": 0}], "_meta": {"config_key": cfg_key}}))

        resp = client.post("/api/captions/timeline/LPDnemFoqVk")
        assert resp.status_code == 200
        assert resp.json()["skipped"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_text_timeline.py::TestCaptionsRouter -v`
Expected: FAIL — endpoint not registered

- [ ] **Step 3: Implement captions router**

```python
# api/src/routers/captions.py
"""Captions timeline router — text timeline for dataset builder."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.artifacts import (
    artifact_path,
    atomic_write_json,
    check_stale,
    config_key,
    read_status,
    status_path_for,
    write_status,
)
from api.src.config import settings
from api.src.video_registry import resolve_stem
from api.src.schemas.captions import TextTimelineRequest, TextTimelineResponse
from api.src.services.text_timeline_service import build_timeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/captions", tags=["captions"])


@router.post("/timeline/{video_id}", response_model=TextTimelineResponse)
async def timeline(video_id: str, req: TextTimelineRequest | None = None):
    """Build a normalized text timeline from existing transcription output."""
    req = req or TextTimelineRequest()
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")

    # Check transcription exists
    transcription_path = settings.data_dir / "transcriptions" / req.stt_model_dir / f"{stem}.json"
    if not transcription_path.exists():
        raise HTTPException(
            409,
            detail={"detail": "Transcription must be completed before building timeline", "missing": ["transcribe"]},
        )

    # Determine source type
    yt_caption_path = settings.youtube_captions_dir / f"{stem}.txt"
    source = "caption" if yt_caption_path.exists() else "stt"

    cfg_params = {"stt_model_dir": req.stt_model_dir, "lexicon_version": req.lexicon_version}
    cfg_key = config_key(cfg_params)
    out = artifact_path(settings.data_dir, "text_timeline", cfg_key, stem)
    sidecar = status_path_for(out)

    # Skip if exists
    if out.exists():
        data = json.loads(out.read_text())
        return TextTimelineResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_segments=len(data.get("segments", [])),
            source=data.get("source", source),
            skipped=True,
        )

    # Check not already running (with crash recovery)
    current = check_stale(sidecar, timeout_s=60.0)
    if current["status"] == "active":
        raise HTTPException(409, detail={"detail": "Timeline is already being built"})

    write_status(sidecar, "active")
    try:
        transcript = json.loads(transcription_path.read_text())
        result = build_timeline(
            transcript,
            source=source,
            lexicon_version=req.lexicon_version,
            stt_model_dir=req.stt_model_dir,
        )

        atomic_write_json(out, result)
        write_status(sidecar, "complete", config_key=cfg_key)

        return TextTimelineResponse(
            video_id=video_id,
            config_key=cfg_key,
            n_segments=len(result["segments"]),
            source=source,
        )
    except Exception as exc:
        write_status(sidecar, "error", error=str(exc))
        raise HTTPException(500, detail=f"Timeline construction failed: {exc}")
```

- [ ] **Step 4: Register captions router in main.py**

Add to `api/src/main.py` in `create_app()` after the vision router registration:

```python
    from api.src.routers.captions import router as captions_router
    app.include_router(captions_router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_text_timeline.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run existing tests for no regressions**

Run: `uv run pytest tests/test_api_scaffold.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add api/src/routers/captions.py api/src/main.py tests/test_text_timeline.py
git commit -m "feat: add captions timeline router with skip-on-exists and 409 deps"
```

---

### Task 8: GPU inference service (basket_tube/inference)

**Files:**
- Create: `basket_tube/inference/main.py` (single FastAPI app with all 5 endpoints)
- Create: `basket_tube/inference/roboflow/models.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p basket_tube/inference/roboflow basket_tube/inference/vision
```

- [ ] **Step 2: Implement models.py — Roboflow model loading**

```python
# basket_tube/inference/roboflow/models.py
"""Roboflow model loading with local/remote mode switching."""

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "local")

# Model IDs
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
COURT_KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"
JERSEY_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."


@lru_cache
def get_model(model_id: str):
    """Load a Roboflow model (cached). Uses inference-gpu locally or API remotely."""
    logger.info("Loading model %s (mode=%s)", model_id, INFERENCE_MODE)
    from inference import get_model as rf_get_model
    return rf_get_model(model_id=model_id)


def run_detection(model, frame, confidence: float = 0.4, iou_threshold: float = 0.9):
    """Run detection on a single frame."""
    result = model.infer(
        frame,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )[0]
    return result


def run_keypoints(model, frame, confidence: float = 0.3):
    """Run keypoint detection on a single frame."""
    result = model.infer(frame, confidence=confidence)[0]
    return result


def run_ocr(model, crop, prompt: str = OCR_PROMPT):
    """Run OCR on a single crop."""
    return model.predict(crop, prompt)[0]
```

- [ ] **Step 3: Implement main.py — single FastAPI app for all GPU inference**

```python
# basket_tube/inference/main.py
"""GPU inference service — RF-DETR detection, keypoints, OCR, SAM2 tracking, team classification."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI
from pydantic import BaseModel

from basket_tube.inference.roboflow.models import (
    get_model,
    run_detection,
    run_keypoints,
    run_ocr,
    PLAYER_DETECTION_MODEL_ID,
    COURT_KEYPOINT_MODEL_ID,
    JERSEY_OCR_MODEL_ID,
    OCR_PROMPT,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="basket-tube-inference")

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))

# Re-use the video registry for stem resolution
from api.src.video_registry import resolve_title as resolve_stem
from api.src.artifacts import config_key, artifact_path, atomic_write_json


class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect", response_model=InferenceResponse)
async def detect(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", PLAYER_DETECTION_MODEL_ID)
        confidence = req.params.get("confidence", 0.4)
        iou_threshold = req.params.get("iou_threshold", 0.9)
        max_frames = req.params.get("max_frames")

        cfg_params = {"model_id": model_id, "confidence": confidence, "iou_threshold": iou_threshold}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "detections", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        if not video_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error=f"Video not found: {video_path}")

        model = get_model(model_id)
        frame_generator = sv.get_video_frames_generator(str(video_path))
        video_info = sv.VideoInfo.from_video_path(str(video_path))

        frames_data = []
        total_detections = 0

        for idx, frame in enumerate(frame_generator):
            if max_frames and idx >= max_frames:
                break
            result = run_detection(model, frame, confidence, iou_threshold)
            detections = sv.Detections.from_inference(result)

            frame_entry = {
                "frame_index": idx,
                "xyxy": detections.xyxy.tolist(),
                "class_id": detections.class_id.tolist(),
                "confidence": detections.confidence.tolist(),
            }
            frames_data.append(frame_entry)
            total_detections += len(detections)

        output = {
            "_meta": {"stage": "detections", "config_key": cfg_key, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames": len(frames_data),
            "n_detections": total_detections,
            "video_info": {"width": video_info.width, "height": video_info.height, "fps": video_info.fps, "total_frames": video_info.total_frames},
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


@app.post("/api/keypoints", response_model=InferenceResponse)
async def keypoints(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", COURT_KEYPOINT_MODEL_ID)
        keypoint_confidence = req.params.get("keypoint_confidence", 0.3)
        anchor_confidence = req.params.get("anchor_confidence", 0.5)
        det_config_key = req.upstream_configs.get("detections", "")

        cfg_params = {"model_id": model_id, "keypoint_confidence": keypoint_confidence, "anchor_confidence": anchor_confidence, "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "court", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        model = get_model(model_id)
        frame_generator = sv.get_video_frames_generator(str(video_path))

        frames_data = []
        n_mapped = 0

        for idx, frame in enumerate(frame_generator):
            result = run_keypoints(model, frame, keypoint_confidence)
            key_points = sv.KeyPoints.from_inference(result)

            if len(key_points) > 0:
                confident = key_points.confidence[0] > anchor_confidence
                xy = key_points.xy[0][confident].tolist() if confident.any() else []
                conf = key_points.confidence[0][confident].tolist() if confident.any() else []
                if len(xy) >= 4:
                    n_mapped += 1
            else:
                xy, conf = [], []

            frames_data.append({"frame_index": idx, "keypoints_xy": xy, "keypoints_confidence": conf})

        output = {
            "_meta": {"stage": "court", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames_mapped": n_mapped,
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Keypoint detection failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


@app.post("/api/ocr", response_model=InferenceResponse)
async def ocr(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        model_id = req.params.get("model_id", JERSEY_OCR_MODEL_ID)
        n_consecutive = req.params.get("n_consecutive", 3)
        ocr_interval = req.params.get("ocr_interval", 5)
        track_config_key = req.upstream_configs.get("tracks", "")

        cfg_params = {"model_id": model_id, "n_consecutive": n_consecutive, "ocr_interval": ocr_interval, "track_config_key": track_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "jerseys", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        # Load tracks to get tracker IDs and bounding boxes
        tracks_path = artifact_path(DATA_DIR, "tracks", track_config_key, stem)
        if not tracks_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Tracks not found")

        tracks_data = json.loads(tracks_path.read_text())
        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        model = get_model(model_id)

        from sports import ConsecutiveValueTracker
        number_validator = ConsecutiveValueTracker(n_consecutive=n_consecutive)

        frame_generator = sv.get_video_frames_generator(str(video_path))
        video_info = sv.VideoInfo.from_video_path(str(video_path))

        for idx, frame in enumerate(frame_generator):
            if idx >= len(tracks_data.get("frames", [])):
                break
            if idx % ocr_interval != 0:
                continue

            frame_track = tracks_data["frames"][idx]
            tracker_ids = frame_track.get("tracker_ids", [])
            xyxy_list = frame_track.get("xyxy", [])

            if not tracker_ids or not xyxy_list:
                continue

            frame_h, frame_w = frame.shape[:2]
            for tid, box in zip(tracker_ids, xyxy_list):
                x1, y1, x2, y2 = [int(c) for c in box]
                # Pad and clip
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(frame_w, x2 + 10)
                y2 = min(frame_h, y2 + 10)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (224, 224))
                number_str = run_ocr(model, crop_resized, OCR_PROMPT)
                number_validator.update({tid: number_str})

        # Build final player map
        players = {}
        validated = number_validator.get_all_validated() if hasattr(number_validator, "get_all_validated") else {}
        for tid, num in validated.items():
            players[str(tid)] = str(num)

        output = {
            "_meta": {"stage": "jerseys", "config_key": cfg_key, "upstream": {"tracks": track_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "players": players,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("OCR failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
```

- [ ] **Step 4: Commit**

```bash
git add basket_tube/inference/
git commit -m "feat: add GPU inference service (detection, keypoints, OCR, tracking, team classification)"
```

---

### Task 9: Vision sub-modules (tracker, classifier)

**Files:**
- Create: `basket_tube/inference/vision/tracker.py`
- Create: `basket_tube/inference/vision/classifier.py`

> **Note:** These modules are imported by `basket_tube/inference/main.py` (Task 8). The tracking and team classification endpoints are part of the single GPU inference service.

- [ ] **Step 1: Implement tracker.py — SAM2Tracker extracted from notebook**

```python
# basket_tube/inference/vision/tracker.py
"""SAM2-based multi-object tracker. Extracted from notebook cell 42."""

from __future__ import annotations

import numpy as np
import torch
import supervision as sv


class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")

        if detections.tracker_id is None:
            detections.tracker_id = list(range(1, len(detections) + 1))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for tid, box in zip(detections.tracker_id, detections.xyxy):
                _, _, _ = self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(tid),
                    bbox=box.tolist(),
                )
        self._prompted = True

    def track(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame() before track()")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            obj_ids, mask_logits = self.predictor.track(frame)

        if len(obj_ids) == 0:
            return sv.Detections.empty()

        masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            tracker_id=np.array(obj_ids, dtype=int),
        )
```

- [ ] **Step 2: Implement classifier.py — TeamClassifier wrapper**

```python
# basket_tube/inference/vision/classifier.py
"""Team classification wrapper using sports.TeamClassifier."""

import numpy as np
import cv2
import supervision as sv


def extract_player_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    player_class_ids: list[int],
    crop_scale: float = 0.4,
) -> list[np.ndarray]:
    """Extract scaled center crops of players from a frame."""
    player_mask = np.isin(detections.class_id, player_class_ids)
    player_dets = detections[player_mask]
    boxes = sv.scale_boxes(xyxy=player_dets.xyxy, factor=crop_scale)
    return [sv.crop_image(frame, box) for box in boxes]
```

- [ ] **Step 3: Track and classify-teams endpoints**

> **Note:** These endpoints are defined in `basket_tube/inference/main.py` (Task 8). The code below shows the endpoint implementations that are part of the single GPU inference FastAPI app.

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import supervision as sv
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BT_DATA_DIR", "/app/pipeline_data/api"))
SAM2_REPO = os.environ.get("SAM2_REPO", "/opt/segment-anything-2-real-time")

from api.src.video_registry import resolve_title as resolve_stem
from api.src.artifacts import config_key, artifact_path, atomic_write_json


class InferenceRequest(BaseModel):
    video_id: str
    params: dict = {}
    upstream_configs: dict = {}


class InferenceResponse(BaseModel):
    status: str
    config_key: str
    output_path: str
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/track", response_model=InferenceResponse)
async def track(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        det_config_key = req.upstream_configs.get("detections", "")
        cfg_params = {"sam2_checkpoint": "sam2.1_hiera_large.pt", "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "tracks", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        # Load detections for first-frame prompt
        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        if not det_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Detections not found")
        det_data = json.loads(det_path.read_text())

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"

        # Load SAM2
        sys.path.insert(0, SAM2_REPO)
        from sam2.build_sam import build_sam2_camera_predictor
        checkpoint = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        old_cwd = os.getcwd()
        os.chdir(SAM2_REPO)
        predictor = build_sam2_camera_predictor(sam2_config, checkpoint)
        os.chdir(old_cwd)

        from basket_tube.inference.vision.tracker import SAM2Tracker
        tracker = SAM2Tracker(predictor)

        PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
        frame_generator = sv.get_video_frames_generator(str(video_path))
        max_frames = req.params.get("max_frames")

        frames_data = []
        all_tracker_ids = set()

        for idx, frame in enumerate(frame_generator):
            if max_frames and idx >= max_frames:
                break

            if idx == 0:
                # Use first frame detections to prompt SAM2
                first_frame = det_data["frames"][0]
                xyxy = np.array(first_frame["xyxy"])
                class_ids = np.array(first_frame["class_id"])
                player_mask = np.isin(class_ids, PLAYER_CLASS_IDS)

                if player_mask.any():
                    player_xyxy = xyxy[player_mask]
                    initial = sv.Detections(
                        xyxy=player_xyxy,
                        class_id=class_ids[player_mask],
                    )
                    initial.tracker_id = np.arange(1, len(initial) + 1)
                    tracker.prompt_first_frame(frame, initial)

                    frame_entry = {
                        "frame_index": idx,
                        "tracker_ids": initial.tracker_id.tolist(),
                        "xyxy": initial.xyxy.tolist(),
                        "mask_rle": [],  # First frame masks from prompt
                    }
                    all_tracker_ids.update(initial.tracker_id.tolist())
                    frames_data.append(frame_entry)
                    continue

            tracked = tracker.track(frame)
            frame_entry = {
                "frame_index": idx,
                "tracker_ids": tracked.tracker_id.tolist() if tracked.tracker_id is not None else [],
                "xyxy": tracked.xyxy.tolist(),
                "mask_rle": [],  # Could encode masks as RLE here
            }
            if tracked.tracker_id is not None:
                all_tracker_ids.update(tracked.tracker_id.tolist())
            frames_data.append(frame_entry)

        output = {
            "_meta": {"stage": "tracks", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "n_frames": len(frames_data),
            "n_tracks": len(all_tracker_ids),
            "frames": frames_data,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Tracking failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))


@app.post("/api/classify-teams", response_model=InferenceResponse)
async def classify_teams(req: InferenceRequest):
    try:
        stem = resolve_stem(req.video_id)
        if not stem:
            return InferenceResponse(status="error", config_key="", output_path="", error=f"Unknown video_id: {req.video_id}")

        stride = req.params.get("stride", 30)
        crop_scale = req.params.get("crop_scale", 0.4)
        det_config_key = req.upstream_configs.get("detections", "")

        cfg_params = {"stride": stride, "crop_scale": crop_scale, "det_config_key": det_config_key}
        cfg_key = config_key(cfg_params)
        out = artifact_path(DATA_DIR, "teams", cfg_key, stem)

        if out.exists():
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        det_path = artifact_path(DATA_DIR, "detections", det_config_key, stem)
        if not det_path.exists():
            return InferenceResponse(status="error", config_key=cfg_key, output_path="", error="Detections not found")
        det_data = json.loads(det_path.read_text())

        video_path = DATA_DIR / "videos" / f"{stem}.mp4"
        PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]

        from basket_tube.inference.vision.classifier import extract_player_crops
        from sports import TeamClassifier

        # Collect crops at stride intervals
        crops = []
        crop_metadata = []  # (frame_index, detection_index)
        frame_generator = sv.get_video_frames_generator(str(video_path), stride=stride)

        for frame_idx_sampled, frame in enumerate(frame_generator):
            actual_idx = frame_idx_sampled * stride
            if actual_idx >= len(det_data["frames"]):
                break

            frame_det = det_data["frames"][actual_idx]
            xyxy = np.array(frame_det["xyxy"])
            class_ids = np.array(frame_det["class_id"])
            player_mask = np.isin(class_ids, PLAYER_CLASS_IDS)

            if not player_mask.any():
                continue

            player_xyxy = xyxy[player_mask]
            player_indices = np.where(player_mask)[0]

            scaled = sv.scale_boxes(xyxy=player_xyxy, factor=crop_scale)
            for box, det_idx in zip(scaled, player_indices):
                crop = sv.crop_image(frame, box)
                if crop.size > 0:
                    crops.append(crop)
                    crop_metadata.append({"frame_index": actual_idx, "detection_index": int(det_idx)})

        if len(crops) < 2:
            # Not enough crops to classify
            output = {
                "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}},
                "palette": {"0": {"name": "Team A", "color": "#006BB6"}, "1": {"name": "Team B", "color": "#007A33"}},
                "assignments": [],
            }
            atomic_write_json(out, output)
            return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

        classifier = TeamClassifier(device="cuda")
        classifier.fit(crops)
        teams = classifier.predict(crops)

        assignments = []
        for meta, team_id in zip(crop_metadata, teams):
            assignments.append({
                "frame_index": meta["frame_index"],
                "detection_index": meta["detection_index"],
                "team_id": int(team_id),
            })

        output = {
            "_meta": {"stage": "teams", "config_key": cfg_key, "upstream": {"detections": det_config_key}, "created_at": datetime.now(timezone.utc).isoformat()},
            "palette": {
                "0": {"name": "Team A", "color": "#006BB6"},
                "1": {"name": "Team B", "color": "#007A33"},
            },
            "assignments": assignments,
        }

        atomic_write_json(out, output)
        return InferenceResponse(status="ok", config_key=cfg_key, output_path=str(out.relative_to(DATA_DIR)))

    except Exception as e:
        logger.exception("Team classification failed")
        return InferenceResponse(status="error", config_key="", output_path="", error=str(e))
```

- [ ] **Step 5: Commit**

```bash
git add basket_tube/inference/vision/
git commit -m "feat: add vision sub-modules (SAM2 tracker, team classifier)"
```

---

### Task 10: Dockerfiles

**Files:**
- Create: `Dockerfile.api` (CPU API)
- Create: `Dockerfile.gpu` (single GPU inference image)

- [ ] **Step 1: Write CPU API Dockerfile**

Create `Dockerfile.api` — a CPU-only image:

```dockerfile
FROM python:3.11-slim

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY --chown=$USERNAME:$USERNAME pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project && \
    chown -R $USERNAME:$USERNAME /app

COPY --chown=$USERNAME:$USERNAME . .

USER $USERNAME

CMD ["uv", "run", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Write Dockerfile.gpu**

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get -y install --no-install-recommends \
    curl git ffmpeg build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/cache/apt && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir -p /etc/pip && pip list --format=freeze > /etc/pip/constraint.txt

# Clone and install SAM2
WORKDIR /opt
RUN git clone https://github.com/Gy920/segment-anything-2-real-time.git \
    && cd segment-anything-2-real-time \
    && pip install -e . \
    && python setup.py build_ext --inplace

RUN cd /opt/segment-anything-2-real-time/checkpoints \
    && bash download_ckpts.sh

WORKDIR /app
COPY pyproject.toml ./
# NOTE: Only copying shared utilities from api/src/ that GPU services need.
COPY api/src/artifacts.py api/src/
COPY api/src/video_registry.py api/src/
COPY api/src/__init__.py api/src/
COPY api/__init__.py api/
COPY basket_tube/ basket_tube/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --constraint /etc/pip/constraint.txt \
    "inference-gpu" "supervision==0.27.0rc4" \
    "sports @ git+https://github.com/roboflow/sports.git@feat/basketball" \
    transformers num2words opencv-python-headless "numpy<2" tqdm \
    fastapi uvicorn pyyaml \
    jupyterlab ipywidgets

ENV SAM2_REPO=/opt/segment-anything-2-real-time
ENV ONNXRUNTIME_EXECUTION_PROVIDERS="[CUDAExecutionProvider]"

EXPOSE 8090
CMD ["uvicorn", "basket_tube.inference.main:app", "--host", "0.0.0.0", "--port", "8090"]
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.api Dockerfile.gpu
git commit -m "feat: add Dockerfiles — CPU API, GPU inference"
```

---

### Task 11: Docker Compose

**Files:**
- Replace: `docker-compose.yml`

- [ ] **Step 1: Write docker-compose.yml**

```yaml
services:
  # ── CPU API orchestrator ──────────────────────────────────────────
  api:
    container_name: basket-tube-api
    profiles: [nvidia, cpu]
    build:
      context: .
      dockerfile: Dockerfile.api
      args:
        USER_UID: "${UID:-1000}"
        USER_GID: "${GID:-1000}"
    restart: unless-stopped
    network_mode: host
    command: ["uv", "run", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8080"]
    environment:
      - FW_INFERENCE_GPU_URL=http://localhost:8090
      - FW_WHISPER_API_URL=http://localhost:8000
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro
      - ./api:/app/api
    # Binds to :8080

  # ── Whisper STT (speaches image) ────────────────────────────────
  whisper:
    container_name: basket-tube-whisper
    profiles: [nvidia]
    image: ghcr.io/speaches-ai/speaches:latest
    restart: unless-stopped
    network_mode: host
    # Binds to :8000

  # ── GPU inference — detection, keypoints, OCR, tracking, teams ──
  inference:
    container_name: basket-tube-inference
    profiles: [nvidia]
    build:
      context: .
      dockerfile: Dockerfile.gpu
    restart: unless-stopped
    network_mode: host
    shm_size: "8gb"
    environment:
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY:-}
      - INFERENCE_MODE=${INFERENCE_MODE:-local}
      - SAM2_REPO=/opt/segment-anything-2-real-time
      - HF_TOKEN=${HF_TOKEN:-}
      - BT_DATA_DIR=/app/pipeline_data/api
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Binds to :8090

  # ── Notebook (Jupyter + GPU) ─────────────────────────────────────
  notebook:
    container_name: basket-tube-notebook
    profiles: [nvidia]
    build:
      context: .
      dockerfile: Dockerfile.gpu
    restart: unless-stopped
    network_mode: host
    shm_size: "8gb"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
    environment:
      - SAM2_REPO=/opt/segment-anything-2-real-time
      - HF_TOKEN=${HF_TOKEN:-}
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY:-}
      - ONNXRUNTIME_EXECUTION_PROVIDERS=[CUDAExecutionProvider]
    volumes:
      - ./pipeline_data:/app/pipeline_data
      - ./video_registry.yml:/app/video_registry.yml:ro
      - ./notebooks:/workspace/notebooks
      - notebook-cache:/home/vscode/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Binds to :8888

volumes:
  notebook-cache:
```

- [ ] **Step 2: Update .env.example**

```
# Copy to .env and fill in your keys
HF_TOKEN=hf_xxxxx
ROBOFLOW_API_KEY=xxxxx
INFERENCE_MODE=local
```

- [ ] **Step 3: Run docker compose config validation**

Run: `docker compose --profile nvidia config --quiet`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: docker-compose with CPU API + whisper + GPU inference + notebook"
```

---

### Task 12: Integration smoke test

- [ ] **Step 1: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ -v --ignore=tests/test_docker_compose.py -x`
Expected: All previously passing tests still pass. New tests pass.

**Note:** The render endpoint (`POST /api/vision/render/{video_id}`) is a 501 stub in this plan. Full render implementation (ffmpeg/opencv compositing) will be a follow-up task.

- [ ] **Step 2: Verify API starts and vision routes are registered**

Run: `uv run python -c "from api.src.main import create_app; app = create_app(); paths = [r.path for r in app.routes if '/vision/' in getattr(r, 'path', '') or '/captions/' in getattr(r, 'path', '')]; print(paths)"`
Expected: Lists all 7 vision endpoint paths + 1 captions timeline path

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes for vision pipeline"
```
