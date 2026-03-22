# BasketTube Action Recognition Training Dataset Design

**Date:** 2026-03-21
**Status:** Draft (rev 1 — aligned with vision pipeline API spec)

## Overview

BasketTube needs action recognition that works from video alone at inference time. Captions, commentary transcripts, and speech-to-text outputs are therefore not part of the runtime model input, but they are valuable during dataset construction as weak supervision for proposing action windows and candidate labels.

This spec defines how BasketTube constructs a training dataset for basketball action recognition from full-game video, aligned text, and vision pipeline artifacts. The output is a reusable, auditable dataset of labeled temporal clips that can train models without any caption dependency at inference time.

## Goals

- Build a video-first training dataset for basketball action recognition
- Use captions/commentary only during dataset construction
- Preserve provenance and confidence for every label
- Support progressive quality tiers: bronze, silver, gold
- Keep the design compatible with the existing BasketTube vision pipeline

## Non-Goals

- Final model architecture selection
- Online inference API design for action recognition
- Full annotation tool implementation details
- Exhaustive basketball ontology from day one

## Core Principle

Text is used to suggest candidate events. Vision is used to ground, validate, refine, or reject those candidates. The final exported training examples must be trainable from video-derived signals alone.

## Data Sources

The dataset builder may use the following inputs:

- Full-game video files
- Official captions when available
- STT transcripts from commentary audio when captions are missing
- Existing or future vision artifacts:
  - player detections
  - player tracks
  - segmentation masks
  - team classification
  - jersey OCR
  - court mapping
  - ball detections or tracks when available
- Optional game metadata:
  - teams
  - competition
  - season
  - scoreboard overlays

## Target Label Set

The initial release should favor a narrow, high-frequency action vocabulary. Recommended phase-1 labels:

- `shot_attempt`
- `three_point_attempt`
- `layup_or_dunk_attempt`
- `made_shot`
- `missed_shot`
- `rebound_offensive`
- `rebound_defensive`
- `assist`
- `steal`
- `block`
- `turnover`
- `free_throw`

Fine-grained play concepts such as `pick_and_roll`, `help_defense_rotation`, or `horns_set` should be deferred until temporal localization and actor attribution are reliable.

## Quality Tiers

Every sample belongs to one of three quality tiers:

- `bronze`: text-only or lightly filtered weak labels
- `silver`: text proposal agrees with vision-derived evidence above threshold
- `gold`: human-reviewed label and temporal boundaries

Training pipelines should support tier-aware sample weighting. Gold data is the primary evaluation source. Silver data is the primary scale source. Bronze data is optional bootstrapping data and should not be mixed uncritically into final evaluation.

## Pipeline

### Stage 1: Source Ingest

- Input one full game at a time
- Assign stable identifiers:
  - `game_id`
  - `video_id`
  - `source_id`
- Normalize:
  - frame rate
  - resolution
  - audio sampling rate
  - timebase
- Persist source metadata:
  - duration
  - fps
  - resolution
  - broadcast source
  - caption availability

### Stage 2: Text Timeline Construction

- Prefer official captions when available
- Otherwise run STT over commentary audio
- Produce timestamped text segments with:
  - `segment_id`
  - `t_start`
  - `t_end`
  - `raw_text`
  - `normalized_text`
  - `source` (`caption` or `stt`)
  - `asr_confidence` when applicable

Normalize basketball vocabulary with a domain lexicon. Example phrase families:

- shot phrases:
  - `for three`
  - `pulls up`
  - `fires`
  - `puts it up`
- make/miss phrases:
  - `knocks it down`
  - `can't get it to go`
- defensive phrases:
  - `blocked`
  - `stripped`
  - `picked off`
- transition phrases:
  - `on the break`
  - `pushes in transition`

Both raw and normalized text must be preserved for auditability.

### Stage 3: Weak Action Proposal Generation

Generate candidate action events from text using a rule-based grammar, and optionally an LLM-assisted normalization pass if introduced later.

Each proposal must include:

- `proposal_id`
- `video_id`
- `label`
- `trigger_text`
- `trigger_segment_id`
- `trigger_timestamp`
- `proposal_window`
- `text_confidence`
- `rule_id`

Examples:

- `for three` -> `three_point_attempt`
- `blocked by` -> `block`
- `comes up with the steal` -> `steal`
- `offensive board` -> `rebound_offensive`

The first implementation should favor precision over recall.

### Stage 4: Temporal Window Expansion

Convert trigger timestamps into candidate clip windows using label-specific priors.

Recommended initial defaults:

- `shot_attempt`: `[-2.0s, +1.5s]`
- `three_point_attempt`: `[-2.5s, +1.5s]`
- `block`: `[-1.5s, +1.0s]`
- `steal`: `[-1.5s, +1.0s]`
- `rebound_*`: `[-1.5s, +1.5s]`
- `assist`: `[-3.0s, +1.5s]`
- `free_throw`: `[-2.0s, +2.0s]`

Store both:

- the original trigger timestamp
- the expanded training window

### Stage 5: Vision Grounding

Run the vision pipeline over the same video and compute visual evidence that supports or rejects the candidate label.

Useful evidence includes:

- player detections and tracks
- player-to-ball proximity
- ball height and arc candidates
- player motion vectors
- court position relative to hoop
- possession change indicators
- defender-attacker interaction near shot release
- rebound scramble patterns after rim contact
- free-throw lane geometry and stationary setup

For each proposal, derive:

- `vision_score`
- `vision_features`
- `grounding_flags`
- `actor_candidates`

Example grounding logic:

- `three_point_attempt` requires shot-like motion and release location consistent with perimeter position
- `block` requires shot contest timing and abrupt ball trajectory interruption near defender proximity
- `steal` requires possession disruption followed by ball control transfer
- `rebound_*` requires a prior shot outcome and ball recovery sequence

### Stage 6: Proposal Resolution

Merge, refine, or reject overlapping proposals.

Rules:

- `three_point_attempt` implies `shot_attempt`
- `made_shot` and `missed_shot` require a shot event in the same window
- `block` may co-occur with `shot_attempt` but changes the shot outcome interpretation
- `assist` should only be emitted when the pass and made basket are temporally linked
- conflicting labels should be marked ambiguous rather than forced when evidence is weak

Outputs:

- final clip boundaries
- final labels
- primary and secondary actors when known
- `combined_confidence`
- `ambiguity_flags`

### Stage 7: Human Review

Human review is required for:

- all evaluation samples
- low-confidence silver candidates
- rare classes
- classes with frequent semantic confusion

Reviewers should confirm:

- label correctness
- temporal boundaries
- primary actor
- relevant secondary actors
- uncertainty notes

Human-reviewed samples are upgraded to `gold`.

### Stage 8: Export

Export training-ready clips and a manifest.

Supported export forms:

- raw RGB clips
- ball-centric crops
- player-centric crops
- track-conditioned clips
- court-projected trajectory features
- fused multimodal training records where all features remain video-derived

## Annotation Rules

Annotation policy must be defined early to avoid drift.

Recommended starting rules:

- `shot_attempt` begins when upward shooting motion starts and ends shortly after release or immediate disruption
- `made_shot` ends when the ball clearly passes through the rim/net sequence
- `missed_shot` ends when the miss outcome is clear and rebound contest begins
- `steal` begins at possession disruption, not at delayed commentary mention
- `assist` includes the pass that directly creates the made shot opportunity
- `block` is assigned when the defense materially alters the shot at or near release

These rules should be versioned. Any future relabeling pass must record the policy version used.

## Dataset Schema

Each exported sample should minimally include:

```json
{
  "sample_id": "uuid",
  "game_id": "game_001",
  "video_id": "vid_001",
  "split": "train",
  "clip_path": "clips/game_001/sample_000123.mp4",
  "fps": 25,
  "t_start": 312.40,
  "t_end": 315.20,
  "labels": ["shot_attempt", "three_point_attempt"],
  "quality_tier": "silver",
  "label_source": "caption+vision",
  "text_evidence": {
    "trigger_text": "for three",
    "timestamp": 313.1,
    "rule_id": "shot.three.generic",
    "source": "caption"
  },
  "vision_evidence": {
    "ball_track_available": true,
    "n_players_tracked": 8,
    "near_hoop": false,
    "court_zone": "left_wing"
  },
  "actors": {
    "primary_tracker_id": "p17",
    "secondary_tracker_ids": ["p22"],
    "team_possession": "home"
  },
  "confidence": {
    "text": 0.81,
    "vision": 0.74,
    "combined": 0.88
  },
  "review": {
    "status": "unreviewed",
    "notes": ""
  },
  "provenance": {
    "pipeline_version": "v0.1",
    "annotation_policy_version": "v0.1",
    "source_video_path": "pipeline_data/api/videos/example.mp4",
    "vision_config_keys": {
      "det_config_key": "c-a3f82b1",
      "track_config_key": "c-b2a91e3",
      "teams_config_key": "c-d4e5f67",
      "court_config_key": "c-1a2b3c4",
      "ball_track_config_key": "c-7e8f9a0"
    },
    "dataset_config_key": "c-ab12cd3",
    "proposal_rule_version": "v0.1"
  }
}
```

## Directory Layout

The dataset workspace follows the vision pipeline's config-key namespacing pattern. Each heuristic/parameter combination produces a distinct config key, enabling reruns and A/B comparisons without overwriting previous results.

```text
pipeline_data/api/analysis/
  action_dataset/
    text_timeline/{config_key}/{stem}.json       # Stage 2
    proposals/{config_key}/{stem}.json            # Stage 3
    proposals/{config_key}/{stem}.status.json     # Status sidecar
    grounded/{config_key}/{stem}.json             # Stage 5
    grounded/{config_key}/{stem}.status.json
    resolved/{config_key}/{stem}.json             # Stage 6
    clips/{config_key}/{game_id}/{sample_id}.mp4  # Stage 8
    features/{config_key}/{sample_id}.npz
    manifests/{config_key}/
      bronze.jsonl
      silver.jsonl
      gold.jsonl
    review/
      queue.jsonl
```

Config keys for dataset stages encode the heuristic version (rule set, thresholds, grounding parameters) and the upstream vision pipeline config keys they depend on.

## Split Strategy

Train/validation/test splits must be done by game, never by clip.

Rules:

- no clips from the same game in multiple splits
- prefer season-aware or broadcast-aware splits when metadata exists
- keep the gold evaluation set fully review-backed

This avoids leakage through repeated camera setups, announcer style, scoreboard overlays, and near-duplicate possessions.

## Hard Negatives

Hard negatives are required and should be explicitly mined.

Examples:

- generic half-court dribbling
- dead-ball periods
- crowd shots
- bench cuts
- commentator excitement without a corresponding action
- aborted drives that do not become shot attempts

These examples are important because text often overstates or lags the true visual event.

## Leakage Controls

To ensure the final recognizer remains caption-independent:

- text fields must never be used as model inputs
- training loaders must not read caption-derived columns
- evaluation should prioritize gold clips with human-verified boundaries
- at least one benchmark subset should be built from no-caption or text-blinded review conditions

## Metrics for Dataset Quality

The dataset construction pipeline should report:

- proposal precision by label
- proposal recall on the reviewed subset
- silver-to-gold promotion rate
- ambiguity rate by label
- per-class sample counts
- per-class temporal boundary adjustment rate during review

These metrics matter more initially than raw dataset size.

## Phased Rollout

### Phase 1

- Implement text timeline creation
- Add weak labeling grammar for 6 to 8 classes
- Export bronze proposals and clips

### Phase 2

- Add vision grounding and confidence fusion
- Export silver dataset
- Start manual review queue

### Phase 3

- Curate a gold validation/test set
- Measure weak-label precision and model performance
- Refine rules using model disagreement and reviewer feedback

## Future Extensions

- scoreboard-aware event validation
- active learning for review prioritization
- semi-supervised relabeling using model confidence
- player-role-conditioned action labels
- possession-level sequence labeling in addition to clip-level classification

## New Code

Recommended future modules:

| File | Purpose |
|---|---|
| `api/src/services/action_dataset_service.py` | Orchestrates dataset construction |
| `api/src/schemas/action_dataset.py` | Dataset, proposal, and review schemas |
| `api/src/routers/action_dataset.py` | API endpoints for build/export/review workflows |
| `notebooks/action_dataset_exploration.ipynb` | Error analysis and heuristic design |

## Success Criteria

The first usable release of the dataset builder should satisfy:

- phase-1 labels have high precision on a reviewed subset
- the gold set is large enough to support model selection
- silver labels measurably improve training scale without collapsing quality
- final action recognition experiments can train and run without any caption input
