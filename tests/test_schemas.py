"""Tests for centralised Pydantic schemas."""

import pytest
from pydantic import ValidationError


class TestDownloadSchemas:
    def test_download_request_valid_url(self):
        from api.src.schemas.download import DownloadRequest
        req = DownloadRequest(url="https://www.youtube.com/watch?v=G3Eup4mfJdA")
        assert req.url == "https://www.youtube.com/watch?v=G3Eup4mfJdA"

    def test_download_request_short_url(self):
        from api.src.schemas.download import DownloadRequest
        req = DownloadRequest(url="https://youtu.be/G3Eup4mfJdA")
        assert req.url.startswith("https://youtu.be/")

    def test_download_request_invalid_url(self):
        from api.src.schemas.download import DownloadRequest
        with pytest.raises(ValidationError, match="Invalid YouTube URL"):
            DownloadRequest(url="not-a-url")

    def test_download_request_missing_url(self):
        from api.src.schemas.download import DownloadRequest
        with pytest.raises(ValidationError):
            DownloadRequest()

    def test_caption_segment_full(self):
        from api.src.schemas.download import CaptionSegment
        seg = CaptionSegment(start=0.0, end=2.5, text="Hello", duration=2.5)
        assert seg.start == 0.0

    def test_caption_segment_optional_fields(self):
        from api.src.schemas.download import CaptionSegment
        seg = CaptionSegment(start=0.0, text="Hello")
        assert seg.end is None

    def test_download_response(self):
        from api.src.schemas.download import DownloadResponse, CaptionSegment
        resp = DownloadResponse(
            video_id="abc123",
            title="My Video",
            caption_segments=[CaptionSegment(start=0.0, end=1.0, text="Hi")],
        )
        assert resp.video_id == "abc123"


class TestTranscribeSchemas:
    def test_transcribe_segment(self):
        from api.src.schemas.transcribe import TranscribeSegment
        seg = TranscribeSegment(start=1.0, end=2.0, text="word")
        assert seg.id is None

    def test_transcribe_response(self):
        from api.src.schemas.transcribe import TranscribeResponse, TranscribeSegment
        resp = TranscribeResponse(
            video_id="vid1",
            language="en",
            text="hello",
            segments=[TranscribeSegment(start=0, end=1, text="hello")],
        )
        assert resp.language == "en"
