"""Tests for api/src/core/artifacts.py — config key, path resolution, status sidecars."""

import json
import time
from pathlib import Path

import pytest

from api.src.artifacts import (
    artifact_path,
    atomic_write_json,
    check_stale,
    config_key,
    read_status,
    status_path_for,
    write_status,
)


class TestConfigKey:
    def test_deterministic(self):
        params = {"model": "yolov8", "conf": 0.5, "iou": 0.45}
        assert config_key(params) == config_key(params)

    def test_order_independent(self):
        params_a = {"model": "yolov8", "conf": 0.5, "iou": 0.45}
        params_b = {"iou": 0.45, "conf": 0.5, "model": "yolov8"}
        assert config_key(params_a) == config_key(params_b)

    def test_format(self):
        key = config_key({"x": 1})
        assert key.startswith("c-")
        assert len(key) == 9  # "c-" + 7 hex chars
        hex_part = key[2:]
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_different_params_give_different_keys(self):
        key_a = config_key({"model": "yolov8"})
        key_b = config_key({"model": "yolov9"})
        assert key_a != key_b

    def test_empty_dict(self):
        key = config_key({})
        assert key.startswith("c-")
        assert len(key) == 9


class TestArtifactPath:
    def test_json_extension(self, tmp_path):
        path = artifact_path(tmp_path, "detect", "c-abc1234", "video1")
        assert path == tmp_path / "analysis" / "detect" / "c-abc1234" / "video1.json"

    def test_mp4_for_renders(self, tmp_path):
        path = artifact_path(tmp_path, "renders", "c-abc1234", "video1")
        assert path == tmp_path / "analysis" / "renders" / "c-abc1234" / "video1.mp4"

    def test_stage_is_directory_component(self, tmp_path):
        path = artifact_path(tmp_path, "track", "c-def5678", "clip42")
        assert path.parts[-4] == "analysis"
        assert path.parts[-3] == "track"
        assert path.parts[-2] == "c-def5678"
        assert path.parts[-1] == "clip42.json"


class TestStatusSidecar:
    def test_status_path_for(self, tmp_path):
        artifact = tmp_path / "analysis" / "detect" / "c-abc1234" / "video1.json"
        sidecar = status_path_for(artifact)
        assert sidecar == tmp_path / "analysis" / "detect" / "c-abc1234" / "video1.status.json"

    def test_write_read_active(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        result = read_status(sidecar)
        assert result["status"] == "active"
        assert "started_at" in result

    def test_write_complete_has_duration_ms(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        write_status(sidecar, "complete")
        result = read_status(sidecar)
        assert result["status"] == "complete"
        assert result["duration_ms"] is not None
        assert isinstance(result["duration_ms"], (int, float))
        assert "completed_at" in result

    def test_write_complete_preserves_started_at(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        active_data = read_status(sidecar)
        started_at = active_data["started_at"]
        write_status(sidecar, "complete")
        complete_data = read_status(sidecar)
        assert complete_data["started_at"] == started_at

    def test_write_error(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "error", error="something went wrong")
        result = read_status(sidecar)
        assert result["status"] == "error"
        assert result["error"] == "something went wrong"

    def test_read_missing_returns_pending(self, tmp_path):
        sidecar = tmp_path / "nonexistent.status.json"
        result = read_status(sidecar)
        assert result == {"status": "pending"}

    def test_read_corrupt_returns_pending(self, tmp_path):
        sidecar = tmp_path / "corrupt.status.json"
        sidecar.write_text("not valid json {{{{")
        result = read_status(sidecar)
        assert result == {"status": "pending"}

    def test_atomic_write_no_tmp_files_linger(self, tmp_path):
        path = tmp_path / "subdir" / "data.json"
        atomic_write_json(path, {"key": "value"})
        # Verify no .tmp files remain
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []
        # Verify the file was written correctly
        assert path.exists()
        assert json.loads(path.read_text()) == {"key": "value"}

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "data.json"
        atomic_write_json(path, {"hello": "world"})
        assert path.exists()

    def test_write_status_with_config_key(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active", config_key="c-abc1234")
        result = read_status(sidecar)
        assert result["config_key"] == "c-abc1234"


class TestCheckStale:
    def test_active_within_timeout_returns_active(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        result = check_stale(sidecar, timeout_s=600.0)
        assert result["status"] == "active"

    def test_stale_active_returns_pending_and_removes_sidecar(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        # Manually backdating started_at to be older than timeout
        data = json.loads(sidecar.read_text())
        data["started_at"] = time.time() - 700.0  # 700s ago, older than 600s timeout
        sidecar.write_text(json.dumps(data))

        result = check_stale(sidecar, timeout_s=600.0)
        assert result == {"status": "pending"}
        assert not sidecar.exists()

    def test_complete_status_unchanged(self, tmp_path):
        sidecar = tmp_path / "status.json"
        write_status(sidecar, "active")
        write_status(sidecar, "complete")
        result = check_stale(sidecar, timeout_s=0.0)  # timeout of 0 — would be stale if active
        assert result["status"] == "complete"

    def test_pending_unchanged(self, tmp_path):
        sidecar = tmp_path / "nonexistent.status.json"
        result = check_stale(sidecar, timeout_s=600.0)
        assert result == {"status": "pending"}


class TestConfigVisionSettings:
    def test_inference_gpu_url_default(self):
        from api.src.config import Settings
        s = Settings()
        assert s.inference_gpu_url == "http://localhost:8090"

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
        assert resolve_stem is resolve_title


class TestResolvedConfig:
    def test_write_resolved_config(self, tmp_path):
        from api.src.artifacts import write_resolved_config
        output_dir = tmp_path / "analysis" / "detections" / "c-abc1234"
        output_dir.mkdir(parents=True)
        write_resolved_config(
            output_dir=output_dir, stage="detections", config_key="c-abc1234",
            params={"confidence": 0.4, "model_id": "test"}, upstream={},
        )
        resolved = output_dir / "config.resolved.json"
        assert resolved.exists()
        import json
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
            output_dir=output_dir, stage="tracks", config_key="c-def5678",
            params={"sam2_checkpoint": "sam2.1_hiera_large.pt"}, upstream={"detections": "c-abc1234"},
        )
        import json
        data = json.loads((output_dir / "config.resolved.json").read_text())
        assert data["upstream"]["detections"] == "c-abc1234"
