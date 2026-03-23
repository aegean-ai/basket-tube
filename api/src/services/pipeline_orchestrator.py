"""Pipeline orchestrator — dependency-aware stage scheduling with SSE broadcast."""

import asyncio
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from api.src.services.event_bus import EventBus
from api.src.services.vision_service import VisionService
from api.src.artifacts import (
    artifact_path, config_key, delete_artifact, read_status,
    status_path_for, write_resolved_config, write_status,
)

try:
    import logfire
except ImportError:
    logfire = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    video_id: str
    bus: EventBus = field(default_factory=EventBus)
    task: asyncio.Task | None = None
    is_active: bool = False
    active_stages: set[str] = field(default_factory=set)
    config_keys: dict[str, str] = field(default_factory=dict)
    stage_states: dict[str, dict] = field(default_factory=dict)


class PipelineOrchestrator:
    def __init__(self, gpu_url: str, data_dir: str | Path):
        self._runs: dict[str, PipelineRun] = {}
        self._gpu_url = gpu_url
        self._data_dir = Path(data_dir)
        self._svc = VisionService(gpu_url=gpu_url)

    def get_or_create_run(self, video_id: str) -> PipelineRun:
        if video_id not in self._runs:
            self._runs[video_id] = PipelineRun(video_id=video_id)
        return self._runs[video_id]

    def start_pipeline(self, video_id: str, settings: dict, stem: str | None = None, from_stage: str | None = None):
        run = self.get_or_create_run(video_id)
        if run.is_active:
            raise RuntimeError(f"Pipeline already running for {video_id}")

        # Reset for new run
        run.bus = EventBus()
        run.is_active = True
        run.active_stages = set()
        run.config_keys = {}
        run.stage_states = {}

        run.task = asyncio.create_task(
            self._execute_pipeline(run, settings, stem or video_id, from_stage)
        )
        return run

    async def cancel_pipeline(self, video_id: str) -> list[str]:
        run = self._runs.get(video_id)
        if not run or not run.is_active:
            return []

        cancelled = list(run.active_stages)
        if run.task:
            run.task.cancel()
        run.is_active = False

        for stage in cancelled:
            await run.bus.emit({
                "event": "stage_error", "stage": stage,
                "error": "cancelled", "timestamp": time.time(),
            })
        return cancelled

    def _read_upstream_config_keys(self, stem: str) -> dict[str, str]:
        """Read config keys from config.resolved.json for from_stage re-runs."""
        keys = {}
        for stage_dir_name in ("detections", "tracks", "teams", "jerseys", "court"):
            stage_dir = self._data_dir / "analysis" / stage_dir_name
            if not stage_dir.exists():
                continue
            for cfg_dir in sorted(stage_dir.iterdir()):
                resolved = cfg_dir / "config.resolved.json"
                if resolved.exists():
                    data = json.loads(resolved.read_text())
                    keys[stage_dir_name] = data.get("config_key", cfg_dir.name)
                    break
        return keys

    async def _execute_pipeline(self, run: PipelineRun, settings: dict, stem: str, from_stage: str | None):
        """Execute the full pipeline DAG."""
        start = time.time()
        span_ctx = logfire.span("pipeline.run", video_id=run.video_id) if logfire else nullcontext()

        try:
            with span_ctx:
                await run.bus.emit({"event": "pipeline_state", "stages": run.stage_states})

                stages_settings = settings.get("stages", {})

                # Handle from_stage: skip stages before it, use existing upstream keys
                skip_stages = set()
                existing_keys = {}
                if from_stage:
                    existing_keys = self._read_upstream_config_keys(stem)
                    stage_order = ["detect", "track", "classify-teams", "court-map", "ocr"]
                    from_idx = stage_order.index(from_stage) if from_stage in stage_order else 0
                    skip_stages = set(stage_order[:from_idx])

                # 1. Download (just mark complete — file already exists)
                await run.bus.emit({"event": "stage_completed", "stage": "download", "timestamp": time.time(), "duration_s": 0})

                # 2. Detect
                if "detect" in skip_stages:
                    det_key = existing_keys.get("detections", "")
                    await run.bus.emit({"event": "stage_skipped", "stage": "detect", "config_key": det_key})
                else:
                    det_params = dict(stages_settings.get("detect", {}))
                    det_key = await self._run_vision_stage(
                        run, "detect", "detections", stem, det_params, upstream={},
                    )

                # 3. Parallel: track + classify-teams + court-map
                track_params = dict(stages_settings.get("track", {}))
                track_params["det_config_key"] = det_key
                teams_params = dict(stages_settings.get("teams", {}))
                teams_params["det_config_key"] = det_key
                court_params = dict(stages_settings.get("court_map", {}))
                court_params["det_config_key"] = det_key

                if "track" in skip_stages:
                    track_key = existing_keys.get("tracks", "")
                    track_task = None
                    await run.bus.emit({"event": "stage_skipped", "stage": "track", "config_key": track_key})
                else:
                    track_task = asyncio.create_task(
                        self._run_vision_stage(run, "track", "tracks", stem, track_params,
                                               upstream={"detections": det_key})
                    )

                teams_task = asyncio.create_task(
                    self._run_vision_stage(run, "classify-teams", "teams", stem, teams_params,
                                           upstream={"detections": det_key})
                )
                court_task = asyncio.create_task(
                    self._run_vision_stage(run, "court-map", "court", stem, court_params,
                                           upstream={"detections": det_key})
                )

                if track_task:
                    track_key = await track_task

                # 4. OCR (needs track)
                ocr_params = dict(stages_settings.get("ocr", {}))
                ocr_params["track_config_key"] = track_key
                ocr_task = asyncio.create_task(
                    self._run_vision_stage(run, "ocr", "jerseys", stem, ocr_params,
                                           upstream={"tracks": track_key})
                )

                await asyncio.gather(teams_task, court_task, ocr_task)

                duration = time.time() - start
                await run.bus.emit({
                    "event": "pipeline_completed",
                    "duration_s": round(duration, 1),
                })
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled for %s", run.video_id)
        except Exception as exc:
            logger.exception("Pipeline failed for %s", run.video_id)
            await run.bus.emit({
                "event": "pipeline_error",
                "error": str(exc),
                "timestamp": time.time(),
            })
        finally:
            run.is_active = False

    async def _run_vision_stage(
        self, run: PipelineRun, stage: str, artifact_stage: str,
        stem: str, params: dict, upstream: dict,
    ) -> str:
        """Run a single vision stage with progress polling."""
        cfg_key = config_key(params)
        run.config_keys[stage] = cfg_key
        out = artifact_path(self._data_dir, artifact_stage, cfg_key, stem)

        # Check cache
        if out.exists():
            await run.bus.emit({"event": "stage_skipped", "stage": stage, "config_key": cfg_key})
            return cfg_key

        await run.bus.emit({"event": "stage_started", "stage": stage, "timestamp": time.time()})
        run.active_stages.add(stage)

        sidecar = status_path_for(out)
        write_status(sidecar, "active", config_key=cfg_key)

        progress_path = out.parent / "_progress.json"
        start = time.time()

        method_map = {
            "detect": self._svc.detect,
            "track": self._svc.track,
            "classify-teams": self._svc.classify_teams,
            "ocr": self._svc.ocr,
            "court-map": self._svc.keypoints,
        }
        method = method_map[stage]

        poller = asyncio.create_task(self._poll_progress(run, stage, progress_path))
        try:
            result = await method(run.video_id, params, upstream_configs=upstream)
        except Exception as exc:
            write_status(sidecar, "error", error=str(exc))
            await run.bus.emit({
                "event": "stage_error", "stage": stage,
                "error": str(exc), "timestamp": time.time(),
            })
            raise
        finally:
            poller.cancel()
            progress_path.unlink(missing_ok=True)
            run.active_stages.discard(stage)

        duration = time.time() - start
        write_status(sidecar, "complete", config_key=cfg_key)
        write_resolved_config(out.parent, stage, cfg_key, params, upstream)

        await run.bus.emit({
            "event": "stage_completed", "stage": stage,
            "config_key": cfg_key, "duration_s": round(duration, 1),
        })
        return cfg_key

    async def _poll_progress(self, run: PipelineRun, stage: str, progress_path: Path):
        """Read progress file every 2s and emit SSE events."""
        try:
            while True:
                await asyncio.sleep(2)
                try:
                    data = json.loads(progress_path.read_text())
                    total = data.get("total_frames", 0)
                    frame = data.get("frame", 0)
                    progress = frame / total if total > 0 else 0
                    await run.bus.emit({
                        "event": "stage_progress", "stage": stage,
                        "progress": round(progress, 3),
                        "frame": frame, "total_frames": total,
                    })
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
        except asyncio.CancelledError:
            pass
