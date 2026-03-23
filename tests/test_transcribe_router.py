"""Tests for POST /api/transcribe/{video_id} endpoint."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def ui_dir(tmp_path):
    """Provide a temporary directory tree."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "transcriptions" / "whisper").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def client(monkeypatch, ui_dir):
    """Test client with settings patched to tmp dir."""
    from api.src.config import settings
    monkeypatch.setattr(settings, "data_dir", ui_dir)

    # Mock resolve_title to return "Test Title" for any video ID
    monkeypatch.setattr(
        "api.src.routers.transcribe.resolve_title",
        lambda vid: "Test Title",
    )

    from api.src.main import app
    with TestClient(app) as c:
        yield c


def _make_whisper_result():
    return {
        "text": "Hello world",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": " Hello world"},
        ],
        "language": "en",
    }


def test_transcribe_returns_segments(client, monkeypatch, ui_dir):
    """POST /api/transcribe/{video_id} returns structured segments."""
    (ui_dir / "videos" / "Test Title.mp4").write_bytes(b"fake-video")

    with patch("api.src.services.whisper_service.transcribe", new=AsyncMock(return_value=_make_whisper_result())):
        resp = client.post("/api/transcribe/G3Eup4mfJdA?use_youtube_captions=false")

    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "G3Eup4mfJdA"
    assert body["language"] == "en"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == " Hello world"


def test_transcribe_saves_json(client, monkeypatch, ui_dir):
    """Transcription result is persisted to transcriptions/whisper/{title}.json."""
    (ui_dir / "videos" / "Test Title.mp4").write_bytes(b"fake-video")

    with patch("api.src.services.whisper_service.transcribe", new=AsyncMock(return_value=_make_whisper_result())):
        client.post("/api/transcribe/G3Eup4mfJdA?use_youtube_captions=false")

    saved = ui_dir / "transcriptions" / "whisper" / "Test Title.json"
    assert saved.exists()
    data = json.loads(saved.read_text())
    assert data["text"] == "Hello world"


def test_transcribe_skips_if_cached(client, monkeypatch, ui_dir):
    """If transcription JSON already exists, don't re-run Whisper."""
    cached = ui_dir / "transcriptions" / "whisper" / "Test Title.json"
    cached.write_text(json.dumps(_make_whisper_result()))

    with patch("api.src.services.whisper_service.transcribe") as mock_transcribe:
        resp = client.post("/api/transcribe/G3Eup4mfJdA")

    assert resp.status_code == 200
    mock_transcribe.assert_not_called()


def test_transcribe_video_not_found(client, monkeypatch, ui_dir):
    """Returns 404 when video ID is not in registry."""
    monkeypatch.setattr(
        "api.src.routers.transcribe.resolve_title",
        lambda vid: None,
    )

    resp = client.post("/api/transcribe/NONEXISTENT")
    assert resp.status_code == 404
