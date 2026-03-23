from api.src.schemas.pipeline import PipelineRunRequest, PipelineRunResponse, PipelineCancelResponse, StageEvent


def test_pipeline_run_request_defaults():
    req = PipelineRunRequest()
    assert req.from_stage is None
    assert req.settings is not None


def test_pipeline_run_response():
    resp = PipelineRunResponse(sse_url="/api/pipeline/events/vid1")
    assert resp.sse_url == "/api/pipeline/events/vid1"


def test_pipeline_cancel_response():
    resp = PipelineCancelResponse(cancelled_stages=["detect", "track"])
    assert len(resp.cancelled_stages) == 2


def test_stage_event_serialization():
    evt = StageEvent(event="stage_completed", stage="detect", config_key="c-abc1234", duration_s=42.1)
    d = evt.model_dump()
    assert d["event"] == "stage_completed"
    assert d["config_key"] == "c-abc1234"
    assert d["duration_s"] == 42.1


def test_stage_event_minimal():
    evt = StageEvent(event="pipeline_completed")
    d = evt.model_dump()
    assert d["stage"] is None
    assert d["progress"] is None
