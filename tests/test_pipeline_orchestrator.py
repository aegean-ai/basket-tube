import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from api.src.services.pipeline_orchestrator import PipelineOrchestrator, PipelineRun


@pytest.mark.asyncio
async def test_get_or_create_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    assert isinstance(run, PipelineRun)
    assert run.video_id == "video1"
    assert orch.get_or_create_run("video1") is run


@pytest.mark.asyncio
async def test_get_or_create_different_videos():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run_a = orch.get_or_create_run("a")
    run_b = orch.get_or_create_run("b")
    assert run_a is not run_b


@pytest.mark.asyncio
async def test_reject_concurrent_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    run.is_active = True
    with pytest.raises(RuntimeError, match="already running"):
        orch.start_pipeline("video1", settings={})


@pytest.mark.asyncio
async def test_cancel_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    mock_task = AsyncMock()
    mock_task.cancel = MagicMock()
    mock_task.cancelled = MagicMock(return_value=False)
    run.task = mock_task
    run.is_active = True
    run.active_stages = {"detect"}

    cancelled = await orch.cancel_pipeline("video1")
    assert "detect" in cancelled
    assert not run.is_active


@pytest.mark.asyncio
async def test_cancel_nonexistent_video():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    cancelled = await orch.cancel_pipeline("nonexistent")
    assert cancelled == []


@pytest.mark.asyncio
async def test_start_pipeline_resets_run():
    orch = PipelineOrchestrator(gpu_url="http://fake:8090", data_dir="/tmp/fake")
    run = orch.get_or_create_run("video1")
    run.config_keys = {"old": "data"}

    # Mock the execute to avoid actual GPU calls
    async def noop(*a, **kw):
        pass
    orch._execute_pipeline = noop

    returned = orch.start_pipeline("video1", settings={}, stem="test")
    assert returned.is_active
    assert returned.config_keys == {}
    assert returned.bus.size == 0
