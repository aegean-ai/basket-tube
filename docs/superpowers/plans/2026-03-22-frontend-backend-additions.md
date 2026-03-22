# Frontend Backend Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three backend capabilities required by the frontend redesign: settings persistence, artifact retrieval, and resolved config snapshots.

**Architecture:** Three small additions to the existing CPU API. Settings persisted as JSON files (no database). Resolved config snapshots written alongside artifacts. Artifact retrieval serves raw JSON for client-side assembly.

**Tech Stack:** Python, FastAPI, Pydantic, existing `atomic_write_json` utility.

**Spec:** `docs/superpowers/specs/2026-03-22-frontend-redesign-design.md` (Configuration Model + Backend Changes Required sections)

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `api/src/schemas/settings.py` | AnalysisSettings, GameContext, AdvancedSettings Pydantic models |
| `api/src/routers/settings.py` | GET/PUT /api/settings/{video_id} |
| `tests/test_settings.py` | Tests for settings persistence |

### Modified Files

| File | Change |
|---|---|
| `api/src/artifacts.py` | Add `write_resolved_config()` |
| `api/src/routers/vision.py` | Add `GET /api/vision/artifacts/{stage}/{video_id}`, call `write_resolved_config()` in each stage |
| `api/src/main.py` | Register settings router |
| `api/src/config.py` | Add `settings_dir` property |

---

### Task 1: Settings schema and persistence

**Files:**
- Create: `api/src/schemas/settings.py`
- Create: `api/src/routers/settings.py`
- Modify: `api/src/config.py`
- Modify: `api/src/main.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_settings.py
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

        settings = {
            "game_context": {"teams": {"0": {"name": "Knicks", "color": "#006BB6"}}, "roster": {"11": "Brunson"}},
            "advanced": {"confidence": 0.3, "iou_threshold": 0.9, "ocr_interval": 5, "crop_scale": 0.4, "stride": 30},
        }
        resp = client.put("/api/settings/LPDnemFoqVk", json=settings)
        assert resp.status_code == 200

        resp = client.get("/api/settings/LPDnemFoqVk")
        assert resp.status_code == 200
        assert resp.json()["game_context"]["teams"]["0"]["name"] == "Knicks"
        assert resp.json()["game_context"]["roster"]["11"] == "Brunson"

    def test_settings_endpoint_registered(self, app):
        paths = list(app.openapi()["paths"].keys())
        assert any("/api/settings" in p for p in paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement settings schema**

```python
# api/src/schemas/settings.py
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
    roster: dict[str, str] = {}  # jersey# -> player name


class AdvancedSettings(BaseModel):
    confidence: float = 0.4
    iou_threshold: float = 0.9
    ocr_interval: int = 5
    crop_scale: float = 0.4
    stride: int = 30


class AnalysisSettings(BaseModel):
    game_context: GameContext = GameContext()
    advanced: AdvancedSettings = AdvancedSettings()
```

- [ ] **Step 4: Add settings_dir to config.py**

Append after `analysis_dir` property in `api/src/config.py`:

```python
    @property
    def settings_dir(self) -> Path:
        return self.data_dir / "settings"
```

- [ ] **Step 5: Implement settings router**

```python
# api/src/routers/settings.py
"""Settings persistence — GET/PUT per-video analysis settings."""

import json
import logging

from fastapi import APIRouter

from api.src.artifacts import atomic_write_json
from api.src.config import settings
from api.src.schemas.settings import AnalysisSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])


def _settings_path(video_id: str):
    return settings.settings_dir / f"{video_id}.json"


@router.get("/settings/{video_id}", response_model=AnalysisSettings)
async def get_settings(video_id: str):
    """Return saved settings for a video, or defaults if not saved."""
    path = _settings_path(video_id)
    if path.exists():
        data = json.loads(path.read_text())
        return AnalysisSettings(**data)
    return AnalysisSettings()


@router.put("/settings/{video_id}", response_model=AnalysisSettings)
async def put_settings(video_id: str, body: AnalysisSettings):
    """Save analysis settings for a video."""
    path = _settings_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, body.model_dump())
    return body
```

- [ ] **Step 6: Register settings router in main.py**

Add after captions router in `api/src/main.py`:

```python
    from api.src.routers.settings import router as settings_router
    app.include_router(settings_router)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_settings.py -v`
Expected: ALL PASSED

- [ ] **Step 8: Commit**

```bash
git add api/src/schemas/settings.py api/src/routers/settings.py api/src/config.py api/src/main.py tests/test_settings.py
git commit -m "feat: add settings persistence API (GET/PUT /api/settings/{video_id})"
```

---

### Task 2: Resolved config snapshots

**Files:**
- Modify: `api/src/artifacts.py`
- Test: `tests/test_artifacts.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_artifacts.py (append)
class TestResolvedConfig:
    def test_write_resolved_config(self, tmp_path):
        from api.src.artifacts import write_resolved_config
        output_dir = tmp_path / "analysis" / "detections" / "c-abc1234"
        output_dir.mkdir(parents=True)

        write_resolved_config(
            output_dir=output_dir,
            stage="detections",
            config_key="c-abc1234",
            params={"confidence": 0.4, "model_id": "test"},
            upstream={},
        )

        resolved = output_dir / "config.resolved.json"
        assert resolved.exists()
        data = json.loads(resolved.read_text())
        assert data["config_key"] == "c-abc1234"
        assert data["stage"] == "detections"
        assert data["params"]["confidence"] == 0.4
        assert data["upstream"] == {}
        assert "resolved_at" in data

    def test_write_resolved_config_with_upstream(self, tmp_path):
        from api.src.artifacts import write_resolved_config
        output_dir = tmp_path / "analysis" / "tracks" / "c-def5678"
        output_dir.mkdir(parents=True)

        write_resolved_config(
            output_dir=output_dir,
            stage="tracks",
            config_key="c-def5678",
            params={"sam2_checkpoint": "sam2.1_hiera_large.pt"},
            upstream={"detections": "c-abc1234"},
        )

        data = json.loads((output_dir / "config.resolved.json").read_text())
        assert data["upstream"]["detections"] == "c-abc1234"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_artifacts.py::TestResolvedConfig -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement write_resolved_config()**

Append to `api/src/artifacts.py`:

```python
def write_resolved_config(
    output_dir: Path,
    stage: str,
    config_key: str,
    params: dict,
    upstream: dict,
) -> None:
    """Write a frozen config snapshot alongside an artifact directory."""
    from datetime import datetime, timezone

    data = {
        "config_key": config_key,
        "stage": stage,
        "params": params,
        "upstream": upstream,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "config.resolved.json", data)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add api/src/artifacts.py tests/test_artifacts.py
git commit -m "feat: add write_resolved_config() for reproducibility snapshots"
```

---

### Task 3: Artifact retrieval endpoint

**Files:**
- Modify: `api/src/routers/vision.py`
- Test: `tests/test_vision_router.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_router.py (append)
class TestArtifactRetrieval:
    def test_get_artifact_unknown_video_returns_404(self, client):
        resp = client.get("/api/vision/artifacts/detections/nonexistent?config_key=c-abc1234")
        assert resp.status_code == 404

    def test_get_artifact_missing_returns_404(self, client, tmp_path, monkeypatch):
        from api.src import config as cfg_mod
        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)
        resp = client.get("/api/vision/artifacts/detections/LPDnemFoqVk?config_key=c-abc1234")
        assert resp.status_code == 404

    def test_get_artifact_returns_json(self, client, tmp_path, monkeypatch):
        import json
        from api.src import config as cfg_mod
        from api.src.artifacts import artifact_path
        monkeypatch.setattr(cfg_mod.settings, "data_dir", tmp_path)

        stem = "Warriors & Lakers Instant Classic - 2021 Play-In Tournament"
        out = artifact_path(tmp_path, "detections", "c-abc1234", stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_frames": 10, "frames": []}))

        resp = client.get("/api/vision/artifacts/detections/LPDnemFoqVk?config_key=c-abc1234")
        assert resp.status_code == 200
        assert resp.json()["n_frames"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vision_router.py::TestArtifactRetrieval -v`
Expected: FAIL — 404 or route not found

- [ ] **Step 3: Add artifact retrieval endpoint to vision router**

Add to `api/src/routers/vision.py`:

```python
@router.get("/artifacts/{stage}/{video_id}")
async def get_artifact(stage: str, video_id: str, config_key: str):
    """Return raw artifact JSON for client-side data assembly."""
    stem = resolve_stem(video_id)
    if stem is None:
        raise HTTPException(404, f"Video '{video_id}' not in registry")

    if stage not in STAGE_NAMES:
        raise HTTPException(404, f"Unknown stage '{stage}'")

    path = artifact_path(settings.data_dir, stage, config_key, stem)
    if not path.exists():
        raise HTTPException(404, f"Artifact not found: {stage}/{config_key}")

    import json
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vision_router.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add api/src/routers/vision.py tests/test_vision_router.py
git commit -m "feat: add GET /api/vision/artifacts/{stage}/{id} for client-side assembly"
```

---

### Task 4: Wire resolved config into vision stage endpoints

**Files:**
- Modify: `api/src/routers/vision.py`

- [ ] **Step 1: Add write_resolved_config import**

Add `write_resolved_config` to the imports from `api.src.artifacts` in `api/src/routers/vision.py`.

- [ ] **Step 2: Add write_resolved_config() call to each stage endpoint**

In each stage endpoint (detect, track, classify-teams, ocr, court-map), after `write_status(sidecar, "complete", ...)`, add:

```python
        write_resolved_config(
            output_dir=out.parent,
            stage="<stage_name>",
            config_key=cfg_key,
            params=cfg_params,
            upstream={...},  # upstream config keys for this stage
        )
```

For detect: `upstream={}`
For track: `upstream={"detections": req.det_config_key}`
For classify-teams: `upstream={"detections": req.det_config_key}`
For ocr: `upstream={"tracks": req.track_config_key}`
For court-map: `upstream={"detections": req.det_config_key}`

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v -q`
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add api/src/routers/vision.py
git commit -m "feat: write config.resolved.json alongside vision stage artifacts"
```

---

### Task 5: Integration verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v -q`
Expected: ALL PASSED, no regressions

- [ ] **Step 2: Verify new endpoints registered**

Run: `uv run python -c "from api.src.main import create_app; app = create_app(); paths = [r.path for r in app.routes if '/settings/' in getattr(r, 'path', '') or '/artifacts/' in getattr(r, 'path', '')]; print(paths)"`
Expected: `['/api/settings/{video_id}', '/api/vision/artifacts/{stage}/{video_id}']`

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes for backend additions"
```
