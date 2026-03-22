# BasketTube Captions API Design

**Date:** 2026-03-21
**Status:** Approved

## Overview

BasketTube needs timestamped text segments from game video for two purposes:

1. **Commentary analysis** — the chat interface that answers natural language questions about the game
2. **Action recognition dataset construction** — weak supervision for proposing and grounding action labels (see `2026-03-21-action-recognition-training-dataset-design.md`, Stage 2)

The captions pipeline already exists in the inherited foreign-whispers codebase. This spec formalizes it, aligns it with BasketTube's config-key namespacing and status sidecar patterns, and defines the text timeline output format required by the dataset spec.

## Caption Source Priority

```
YouTube captions available?
  ├── yes → use them (fastest, highest quality for broadcast games)
  └── no  → run Whisper STT over commentary audio (GPU, slower)
```

This matches the existing `use_youtube_captions` flag behavior. YouTube captions are preferred because they are human-edited for broadcast games. Whisper STT is the fallback for games without official captions.

## Architecture

The captions pipeline runs entirely on the **CPU API container** (:8080). It does not require GPU inference services. Whisper runs locally (loaded lazily on first use) or is delegated to a remote Whisper server (configurable via `FW_WHISPER_BACKEND`).

```
CPU API (:8080)
  POST /api/download          → video + YouTube captions
  POST /api/transcribe/{id}   → Whisper STT or YouTube caption conversion
  POST /api/captions/timeline/{id}  → text timeline for dataset builder (NEW)
```

The first two endpoints already exist. The third is new — it produces the structured text timeline artifact that the dataset spec's Stage 2 requires.

## Existing Endpoints (no changes)

### POST /api/download

- Downloads video MP4 to `pipeline_data/api/videos/{stem}.mp4`
- Downloads YouTube captions to `pipeline_data/api/youtube_captions/{stem}.txt` (line-delimited JSON)
- Returns `video_id`, `title`, caption segments
- Skip-on-exists: skips download if files already present

### POST /api/transcribe/{video_id}

- Reads YouTube captions if available and `use_youtube_captions=True` (default)
- Falls back to Whisper STT on the video file
- Writes result to `pipeline_data/api/transcriptions/{stt_model_dir}/{stem}.json`
- Returns `video_id`, `language`, `text`, `segments`, `skipped`
- Existing output format (Whisper-compatible):

```json
{
  "language": "en",
  "text": "full transcript text...",
  "segments": [
    {"id": 0, "start": 12.5, "end": 15.2, "text": "Curry for three..."}
  ]
}
```

## New Endpoint: Text Timeline

### POST /api/captions/timeline/{video_id}

Produces the structured text timeline artifact required by the action recognition dataset spec (Stage 2). This is a CPU-side transformation of the existing transcription output — no new models.

- **Input:** `pipeline_data/api/transcriptions/{stt_model_dir}/{stem}.json` (from transcribe endpoint)
- **Output:** `pipeline_data/api/analysis/text_timeline/{config_key}/{stem}.json`
- **Config key inputs:** `{stt_model_dir, source_type, lexicon_version}`
- **Dependencies:** Transcription must exist (409 if missing)

#### Request

```python
class TextTimelineRequest(BaseModel):
    stt_model_dir: str = "whisper"          # which transcription to use
    lexicon_version: str = "v0.1"           # basketball vocabulary normalization version
```

#### Response

```python
class TextTimelineResponse(BaseModel):
    video_id: str
    config_key: str
    n_segments: int
    source: str                              # "caption" or "stt"
    skipped: bool = False
```

#### Output JSON Schema

```json
{
  "_meta": {
    "stage": "text_timeline",
    "config_key": "c-f1a2b3c",
    "upstream": {
      "transcription": "whisper"
    },
    "created_at": "2026-03-21T14:30:00Z"
  },
  "source": "caption",
  "lexicon_version": "v0.1",
  "segments": [
    {
      "segment_id": 0,
      "t_start": 12.50,
      "t_end": 15.20,
      "raw_text": "Curry for three!",
      "normalized_text": "curry for three",
      "source": "caption",
      "asr_confidence": null
    },
    {
      "segment_id": 1,
      "t_start": 15.20,
      "t_end": 17.80,
      "raw_text": "He knocks it down!",
      "normalized_text": "he knocks it down",
      "source": "caption",
      "asr_confidence": null
    }
  ]
}
```

#### Normalization Rules

The text timeline normalizes raw text for downstream pattern matching:

1. **Case:** lowercase all text
2. **Punctuation:** strip trailing punctuation, preserve apostrophes
3. **Domain lexicon:** normalize basketball-specific phrases:
   - `"3"` / `"three pointer"` / `"trey"` → `"three"`
   - `"dunk"` / `"slam"` / `"jam"` → `"dunk"`
   - `"layup"` / `"lay-up"` / `"lay up"` → `"layup"`
   - `"and-one"` / `"and one"` / `"and 1"` → `"and one"`
4. **Source tagging:** each segment records whether it came from `"caption"` or `"stt"`
5. **ASR confidence:** populated only for STT segments (Whisper provides per-segment log probability)

Both `raw_text` and `normalized_text` are preserved for auditability, as required by the dataset spec.

## Caching & Status

All endpoints follow the config-key namespacing and status sidecar patterns from the vision pipeline spec:

- Text timeline output is namespaced by `{config_key}` (hash of `stt_model_dir` + `lexicon_version`)
- Status sidecar at `text_timeline/{config_key}/{stem}.status.json`
- Skip-on-exists with `skipped: true`
- Atomic writes (`.tmp` + rename)
- Crash recovery via `check_stale()`

The existing download and transcribe endpoints predate this pattern and keep their flat directory layout (`youtube_captions/{stem}.txt`, `transcriptions/whisper/{stem}.json`). They are already model-namespaced via `stt_model_dir` which serves the same purpose.

## Configuration

No new settings required. The existing settings cover all captions needs:

| Setting | Env Var | Default | Purpose |
|---|---|---|---|
| `whisper_model` | `FW_WHISPER_MODEL` | `base` | Whisper model size |
| `whisper_backend` | `FW_WHISPER_BACKEND` | `local` | `local` or `remote` |
| `whisper_api_url` | `FW_WHISPER_API_URL` | `http://localhost:8000` | Remote Whisper endpoint |
| `stt_model_dir` | `FW_STT_MODEL_DIR` | `whisper` | Transcription output subdirectory |

## Docker

The captions pipeline runs on the CPU API container. No GPU container changes needed. The existing Whisper-GPU container (`speaches`) can be re-added to `docker-compose.yml` under the `nvidia` profile when remote STT is needed for large-scale processing.

## New Code

| File | Purpose |
|---|---|
| `api/src/routers/captions.py` | New timeline endpoint (reuses existing download/transcribe routers) |
| `api/src/schemas/captions.py` | `TextTimelineRequest`, `TextTimelineResponse` |
| `api/src/services/text_timeline_service.py` | Normalization logic, lexicon, segment transformation |
| `tests/test_text_timeline.py` | Tests for normalization, timeline construction |

## Existing Code Unchanged

| File | Status |
|---|---|
| `api/src/routers/download.py` | Unchanged — still handles video + caption download |
| `api/src/routers/transcribe.py` | Unchanged — still handles Whisper/YouTube caption transcription |
| `api/src/services/download_service.py` | Unchanged |
| `api/src/services/transcription_service.py` | Unchanged |

## Relationship to Dataset Spec

The text timeline output directly feeds the dataset spec's Stage 2 (Text Timeline Construction) and Stage 3 (Weak Action Proposal Generation). The proposal generator reads `text_timeline/{config_key}/{stem}.json` and applies pattern matching rules to produce action candidates.

```
download → transcribe → text_timeline → [dataset: proposals → grounding → ...]
                                ↑
                          This spec's new endpoint
```
