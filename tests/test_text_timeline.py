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
