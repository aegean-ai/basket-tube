# BasketTube Frontend Redesign

**Date:** 2026-03-22
**Status:** Approved

## Overview

Replace the foreign-whispers dubbing studio frontend with a basketball game analysis interface. The UI provides a video player, vision pipeline controls, player roster, court visualization, and a chat shell — all wired to the BasketTube API.

Same tech stack: Next.js 16, React 19, shadcn/ui, Tailwind v4, dark theme.

## Layout

Sidebar + Video + Tabbed Panels:

```
┌──────┬────────────────────────────────────────┐
│      │                                        │
│  V   │          Video Player                  │
│  i   │      (annotated overlay when           │
│  d   │          results available)             │
│  e   │                                        │
│  o   ├────────────────────────────────────────┤
│      │ [Pipeline] [Players] [Court] [Chat]    │
│  L   │                                        │
│  i   │    Selected tab content renders here   │
│  s   │                                        │
│  t   │                                        │
│      │                                        │
├──────┤                                        │
│ [⚙]  │                                        │
│[▶ An]│                                        │
└──────┴────────────────────────────────────────┘
```

- **Left sidebar:** video list from `video_registry.yml`, "Analyze" button, settings gear icon
- **Top:** large video player. Shows source video. (Annotated render is a future feature — the render stage is a 501 stub.)
- **Bottom:** tabbed panel with 4 tabs

## Tabs

### Pipeline Tab

Stage progress table with 8 rows:

| Stage | Description | Status | Duration | Actions |
|-------|-------------|--------|----------|---------|
| Download | Fetch video from YouTube | Done | 12s | |
| Transcribe | Speech-to-text (Whisper) | Done | 45s | |
| Detect | Player detection (RF-DETR) | Running | 2m 15s | |
| Track | Player tracking (SAM2) | Pending | — | |
| Classify Teams | Team assignment (SigLIP) | Pending | — | |
| OCR | Jersey number recognition | Pending | — | |
| Court Map | Court keypoint mapping | Pending | — | |
| Render | Annotated video output | Pending | — | (stub) |

Status badges: Running (blue pulse), Done (green), Skipped (gray), Error (red), Pending (dim).

Each completed stage shows a "Re-run" button. Re-running a stage invalidates all downstream stages.

### Players Tab

Roster table populated from OCR + team classification + track artifacts:

| # | Player | Team | First Seen | |
|---|--------|------|------------|--|
| 30 | Stephen Curry | Warriors | 0:04 | → (click seeks video) |
| 23 | LeBron James | Lakers | 0:02 | → |

- Rows colored by team (using palette from classify-teams artifact)
- Click a row → video player seeks to that player's `first_frame_time`
- Player names resolved from roster mapping in settings (jersey# → name)
- If OCR hasn't run yet, table shows "Run pipeline to detect players"

**Data assembly:** The frontend fetches three artifacts to build the table:

1. **OCR artifact** (`jerseys/{config}/{stem}.json`): `players` map of `tracker_id → jersey_number`
2. **Teams artifact** (`teams/{config}/{stem}.json`): `assignments` array of `{frame_index, detection_index, team_id}` + `palette`
3. **Tracks artifact** (`tracks/{config}/{stem}.json`): `frames` array with per-frame `tracker_ids` and `xyxy` boxes

The frontend joins these: for each `tracker_id` in the OCR players map, find its team from the assignments (via the track→detection index mapping), and find its first frame appearance from the tracks data. The seek target is `first_frame_index / fps` (fps from the detection artifact's `video_info`).

**Artifact retrieval:** A new API endpoint serves artifact JSON:

```
GET /api/vision/artifacts/{stage}/{video_id}?config_key={key}
```

Returns the raw JSON artifact file for client-side assembly. This avoids baking presentation logic into the API — the frontend reads the structured data and builds the table itself.

### Court Tab

Placeholder for bird's-eye court visualization.

Shows a static message: "Court visualization coming soon" with the NBA court diagram background. Will eventually render player positions and trajectories from the court-map stage output.

### Chat Tab

Message UI shell with no backend:

- Message list (scrollable)
- Text input + send button at bottom
- Send button shows a toast: "Chat backend coming soon"
- UI is fully functional (messages appear in list) but responses are a static placeholder

## Pipeline Orchestration

### Analyze Button

The sidebar "Analyze" button triggers `runPipeline(video, settings)`:

```
1. download       — POST /api/download (if not cached)
2. transcribe     — POST /api/transcribe/{id} (YouTube captions or Whisper)
3. detect         — POST /api/vision/detect/{id}
4. track          — POST /api/vision/track/{id}          (needs detect config_key)
5. ocr            — POST /api/vision/ocr/{id}            (needs track config_key)
6. classify-teams — POST /api/vision/classify-teams/{id}  (needs detect config_key)
7. court-map      — POST /api/vision/court-map/{id}      (needs detect config_key)
8. render         — POST /api/vision/render/{id}          (needs all 5 vision config_keys, stub for now)
```

Stages 3-8 execute sequentially (single GPU constraint). Each stage returns a `config_key` that downstream stages need. Render is currently a 501 stub — the stage appears in the table but is skipped automatically until implemented.

### Config Key Chaining

The pipeline hook stores config keys returned by each stage:

```typescript
config_keys: {
  detect: "c-a3f82b1",      // from detect response
  track: "c-b2a91e3",       // from track response (encodes detect key)
  "classify-teams": "c-...",
  ocr: "c-...",
  "court-map": "c-...",
}
```

When calling `track`, the hook passes `det_config_key: config_keys.detect`. When calling `ocr`, it passes `track_config_key: config_keys.track`.

### Stage Re-run

Each completed stage has a "Re-run" button. Re-running works through the existing config-key mechanism — no backend invalidation endpoint is needed:

1. **If parameters changed** (e.g., user adjusts confidence in Advanced Settings): the new parameters produce a new config key → the API treats it as a fresh run and processes from scratch. Old artifacts remain on disk under their config key.

2. **If parameters unchanged** (user clicks Re-run without changing settings): the API returns `skipped: true` because the output already exists. To force reprocessing with identical parameters, the frontend must delete the cached artifact first. This is not exposed in the UI for now — re-run is only useful after changing settings.

3. **Downstream invalidation is frontend-only**: the pipeline hook clears `config_keys` and `stages` state for downstream stages in the reducer. It then re-executes the stage and all downstream stages sequentially with the new upstream config keys. The API naturally produces new outputs because the upstream config key changed.

Example: user changes detection confidence from 0.4 to 0.3:
- Detect produces `c-7e19d04` (new config key, different from old `c-a3f82b1`)
- Track receives `det_config_key: "c-7e19d04"` → new track config key → fresh processing
- All downstream stages cascade with new keys

### Dependency Map

```
detect
  ├── track ──── ocr
  ├── classify-teams
  └── court-map
```

## Settings Dialog

Two sections:

### Game Context (always visible)

| Setting | Type | Default |
|---|---|---|
| Team 0 Name | text input | "Team A" |
| Team 0 Color | color picker | #006BB6 |
| Team 1 Name | text input | "Team B" |
| Team 1 Color | color picker | #007A33 |
| Roster | key-value editor | {} |

Roster editor: table with jersey# (input) → player name (input), add/remove rows. Pre-populated from `TEAM_ROSTERS` constant if available.

### Advanced (behind toggle)

| Setting | Type | Default | Description |
|---|---|---|---|
| Detection Confidence | slider 0-1 | 0.4 | RF-DETR confidence threshold |
| IOU Threshold | slider 0-1 | 0.9 | Non-max suppression overlap |
| OCR Interval | number | 5 | Run OCR every N frames |
| Team Crop Scale | slider 0-1 | 0.4 | Center crop scale for team classification |
| Sampling Stride | number | 30 | Frame sampling stride for team classification |

Advanced settings map directly to API request parameters (`DetectRequest`, `ClassifyTeamsRequest`, `OCRRequest`). Parameters not exposed here (e.g., `model_id`, `max_frames`, `n_consecutive`, `keypoint_confidence`, `anchor_confidence`) use server defaults and are not surfaced in the UI.

## Types

### `lib/types.ts`

```typescript
// Video
interface Video { id: string; title: string; url: string; }

// Vision pipeline
type VisionStage = "download" | "transcribe" | "detect" | "track" | "ocr" | "classify-teams" | "court-map" | "render";
type StageStatus = "pending" | "active" | "complete" | "skipped" | "error";

interface StageState {
  status: StageStatus;
  config_key?: string;
  result?: unknown;
  error?: string;
  duration_ms?: number;
  started_at?: number;
}

interface PipelineState {
  status: "idle" | "running" | "complete" | "error";
  stages: Record<VisionStage, StageState>;
  videoId?: string;
}

// UI state — managed by analysis-layout, not the pipeline hook
type TabId = "pipeline" | "players" | "court" | "chat";

// Stage responses
interface DetectResponse { video_id: string; config_key: string; n_frames: number; n_detections: number; skipped: boolean; }
interface TrackResponse { video_id: string; config_key: string; n_frames: number; n_tracks: number; skipped: boolean; }
interface ClassifyTeamsResponse { video_id: string; config_key: string; palette: Record<string, {name: string; color: string}>; skipped: boolean; }
interface OCRResponse { video_id: string; config_key: string; players: Record<string, string>; skipped: boolean; }
interface CourtMapResponse { video_id: string; config_key: string; n_frames_mapped: number; skipped: boolean; }
interface RenderResponse { video_id: string; config_key: string; skipped: boolean; }
interface PipelineStatusResponse { video_id: string; stages: Record<string, StageStatusResponse>; }
interface StageStatusResponse { status: string; config_key?: string; started_at?: string; completed_at?: string; duration_ms?: number; error?: string; }

// Settings
interface GameContext {
  teams: Record<string, { name: string; color: string }>;
  roster: Record<string, string>;  // jersey# -> player name (team derived from classify-teams output)
}

interface AdvancedSettings {
  confidence: number;
  iou_threshold: number;
  ocr_interval: number;
  crop_scale: number;
  stride: number;
}

interface AnalysisSettings {
  gameContext: GameContext;
  advanced: AdvancedSettings;
}

// Captions timeline (see captions API spec: 2026-03-21-captions-api-design.md)
interface TextTimelineRequest { stt_model_dir?: string; lexicon_version?: string; }
interface TextTimelineResponse { video_id: string; config_key: string; n_segments: number; source: string; skipped: boolean; }

// Chat
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
```

## API Client

### `lib/api.ts`

```typescript
// Kept from foreign-whispers
downloadVideo(url: string): Promise<DownloadResponse>
transcribeVideo(videoId: string, useYoutubeCaptions?: boolean): Promise<TranscribeResponse>

// New — vision pipeline
detectPlayers(videoId: string, params?: DetectRequest): Promise<DetectResponse>
trackPlayers(videoId: string, req: TrackRequest): Promise<TrackResponse>
classifyTeams(videoId: string, req: ClassifyTeamsRequest): Promise<ClassifyTeamsResponse>
ocrJerseys(videoId: string, req: OCRRequest): Promise<OCRResponse>
mapCourt(videoId: string, req: CourtMapRequest): Promise<CourtMapResponse>
renderVideo(videoId: string, req: RenderRequest): Promise<RenderResponse>
getPipelineStatus(videoId: string): Promise<PipelineStatusResponse>
getArtifact(stage: string, videoId: string, configKey: string): Promise<any>  // GET /api/vision/artifacts/{stage}/{id}?config_key={key}

// New — captions
buildTimeline(videoId: string, req?: TextTimelineRequest): Promise<TextTimelineResponse>

// URL helpers
getVideoUrl(videoId: string): string                    // original source video
// getAnnotatedVideoUrl() — deferred until render stage is implemented
```

## Component Inventory

### Delete (foreign-whispers specific)

| File | Reason |
|---|---|
| `transcript-view.tsx` | Dubbing-specific transcript viewer |
| `translation-view.tsx` | Bilingual comparison (no translation in BasketTube) |
| `audio-player.tsx` | No standalone audio playback needed |
| `dubbing-method-accordion.tsx` | Dubbing alignment selector |
| `diarization-accordion.tsx` | Speaker diarization selector |
| `voice-cloning-accordion.tsx` | Voice cloning selector |
| `lib/config-id.ts` | Dubbing variant generator |
| `contexts/studio-settings-context.tsx` | Replaced by analysis-settings-context |

### Rewrite

| File | Change |
|---|---|
| `app-sidebar.tsx` | Rebrand to "BasketTube", remove dubbing refs, add Analyze button |
| `studio-layout.tsx` → `analysis-layout.tsx` | New layout with tabbed panels |
| `pipeline-table.tsx` | New vision stages, re-run buttons |
| `pipeline-cards.tsx` | New metrics (players, teams, frames, court frames) |
| `pipeline-status-bar.tsx` | New stage names |
| `settings-dialog.tsx` | Game context + advanced vision parameters |
| `hooks/use-pipeline.ts` | New stage sequence, config key chaining, re-run logic |
| `lib/types.ts` | All new types (see above) |
| `lib/api.ts` | New API client (see above) |
| `video-canvas.tsx` | Full rewrite — remove variant/subtitle logic, show source video only |
| `app/layout.tsx` | Update metadata, swap settings provider |

### Create New

| File | Purpose |
|---|---|
| `players-table.tsx` | Roster table with team colors, click-to-seek |
| `court-view.tsx` | Placeholder court visualization |
| `chat-panel.tsx` | Chat UI shell (message list + input, placeholder backend) |
| `analysis-tabs.tsx` | Tab container (Pipeline \| Players \| Court \| Chat) |
| `contexts/analysis-settings-context.tsx` | GameContext + AdvancedSettings state provider |

### Keep As-Is

| File | Reason |
|---|---|
| All `ui/` components | Standard shadcn design system |
| `video-player.tsx` | Generic video player |
| `hooks/use-elapsed.ts` | Timer utility |
| `hooks/use-mobile.ts` | Responsive breakpoint |
| `lib/utils.ts` | Class utilities |

## Data Flow

```
page.tsx (Server)
  ↓ fetches GET /api/videos
  ↓ passes videos[]
analysis-layout.tsx (Root Client)
  ↓ manages selectedVideoId, selectedTab
  ├→ AppSidebar (video list + Analyze button + settings gear)
  ├→ VideoPlayer (source or annotated video)
  ├→ AnalysisTabs
  │   ├→ PipelineTable (stage progress + re-run)
  │   ├→ PlayersTable (roster + click-to-seek)
  │   ├→ CourtView (placeholder)
  │   └→ ChatPanel (UI shell)
  └→ PipelineStatusBar (live status)

usePipeline hook
  ↓ runPipeline() calls API stages sequentially
  ↓ stores config_keys from each response
  ↓ tracks stages[].status, duration_ms, error

useAnalysisSettings hook
  ↓ gameContext: team names, colors, rosters
  ↓ advanced: confidence, iou, ocr_interval, stride
```

## Implementation Notes

- Use shadcn MCP (https://ui.shadcn.com/docs/mcp) during implementation for component code
- The existing dark theme and color palette are kept — only the golden accent (#f0c040) and component content change
- `page.tsx` server component pattern stays the same (fetch videos, pass to client layout)
- `video-canvas.tsx` is a full rewrite (current component is built around dubbed variants and subtitle switching)
- Video player shows source video only — annotated render deferred until render stage is implemented
- The `public/demo/` directory with foreign-whispers demo data can be deleted
