# Frontend Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the foreign-whispers dubbing studio frontend with a basketball game analysis interface — video player, vision pipeline controls, player roster, court placeholder, and chat shell.

**Architecture:** Same Next.js 16 + React 19 + shadcn/ui + Tailwind v4 stack. Layout: sidebar (video list + Analyze + settings) | video player (top) | tabbed panels (Pipeline | Players | Court | Chat). Pipeline hook orchestrates 8 stages sequentially, passing config keys between dependent stages. Settings persisted via the backend API.

**Tech Stack:** Next.js 16, React 19, TypeScript, shadcn/ui, Tailwind CSS v4, Lucide icons.

**Spec:** `docs/superpowers/specs/2026-03-22-frontend-redesign-design.md`

**Prerequisite:** `docs/superpowers/plans/2026-03-22-frontend-backend-additions.md` must be completed first.

**Note:** Use shadcn MCP (https://ui.shadcn.com/docs/mcp) for component code during implementation.

---

## File Structure

### Delete

| File | Reason |
|---|---|
| `src/components/transcript-view.tsx` | Dubbing-specific |
| `src/components/translation-view.tsx` | Dubbing-specific |
| `src/components/audio-player.tsx` | Not needed |
| `src/components/dubbing-method-accordion.tsx` | Dubbing-specific |
| `src/components/diarization-accordion.tsx` | Dubbing-specific |
| `src/components/voice-cloning-accordion.tsx` | Dubbing-specific |
| `src/lib/config-id.ts` | Dubbing variant generator |
| `src/contexts/studio-settings-context.tsx` | Replaced |
| `src/components/studio-layout.tsx` | Replaced by analysis-layout |
| `public/demo/` | Foreign-whispers demo data |

### Rewrite

| File | Change |
|---|---|
| `src/lib/types.ts` | All new types for vision pipeline |
| `src/lib/api.ts` | New API client functions |
| `src/hooks/use-pipeline.ts` | New 8-stage pipeline with config key chaining |
| `src/components/app-sidebar.tsx` | Rebrand, Analyze button, settings gear |
| `src/components/pipeline-table.tsx` | 8 vision stages, re-run buttons |
| `src/components/pipeline-cards.tsx` | New metrics |
| `src/components/pipeline-status-bar.tsx` | New stage names |
| `src/components/video-canvas.tsx` | Remove variants, show source video |
| `src/components/settings-dialog.tsx` | Game context + advanced params |
| `src/app/layout.tsx` | Update metadata, swap provider |
| `src/app/page.tsx` | Import AnalysisLayout instead of StudioLayout |

### Create

| File | Purpose |
|---|---|
| `src/components/analysis-layout.tsx` | Root layout: sidebar + video + tabs |
| `src/components/analysis-tabs.tsx` | Tab container (Pipeline \| Players \| Court \| Chat) |
| `src/components/players-table.tsx` | Roster table, team colors, click-to-seek |
| `src/components/court-view.tsx` | Placeholder court visualization |
| `src/components/chat-panel.tsx` | Chat UI shell |
| `src/contexts/analysis-settings-context.tsx` | Settings state + persistence via API |

---

### Task 1: Delete foreign-whispers components and clean up

**Files:**
- Delete: 8 component files + 1 lib file + 1 context + `public/demo/`

- [ ] **Step 1: Delete foreign-whispers-specific files**

```bash
rm frontend/src/components/transcript-view.tsx
rm frontend/src/components/translation-view.tsx
rm frontend/src/components/audio-player.tsx
rm frontend/src/components/dubbing-method-accordion.tsx
rm frontend/src/components/diarization-accordion.tsx
rm frontend/src/components/voice-cloning-accordion.tsx
rm frontend/src/lib/config-id.ts
rm -rf frontend/public/demo
# NOTE: studio-layout.tsx and studio-settings-context.tsx are deleted in Task 5
# when their replacements (analysis-layout, analysis-settings-context) are ready.
# This keeps the build from breaking between tasks.
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore: delete foreign-whispers frontend components"
```

---

### Task 2: Types and API client

**Files:**
- Rewrite: `src/lib/types.ts`
- Rewrite: `src/lib/api.ts`

- [ ] **Step 1: Rewrite types.ts**

```typescript
// src/lib/types.ts

// Video
export interface Video {
  id: string;
  title: string;
  url: string;
}

// Vision pipeline stages
export type VisionStage =
  | "download"
  | "transcribe"
  | "detect"
  | "track"
  | "ocr"
  | "classify-teams"
  | "court-map"
  | "render";

export type StageStatus = "pending" | "active" | "complete" | "skipped" | "error";

export interface StageState {
  status: StageStatus;
  config_key?: string;
  result?: unknown;
  error?: string;
  duration_ms?: number;
  started_at?: number;
}

export interface PipelineState {
  status: "idle" | "running" | "complete" | "error";
  stages: Record<VisionStage, StageState>;
  videoId?: string;
}

// UI state
export type TabId = "pipeline" | "players" | "court" | "chat";

// API responses
export interface DownloadResponse {
  video_id: string;
  title: string;
  caption_segments: { start: number; end: number; text: string }[];
}

export interface TranscribeResponse {
  video_id: string;
  language: string;
  text: string;
  segments: { id?: number; start: number; end: number; text: string }[];
  skipped: boolean;
}

export interface DetectResponse {
  video_id: string;
  config_key: string;
  n_frames: number;
  n_detections: number;
  skipped: boolean;
}

export interface TrackResponse {
  video_id: string;
  config_key: string;
  n_frames: number;
  n_tracks: number;
  skipped: boolean;
}

export interface ClassifyTeamsResponse {
  video_id: string;
  config_key: string;
  palette: Record<string, { name: string; color: string }>;
  skipped: boolean;
}

export interface OCRResponse {
  video_id: string;
  config_key: string;
  players: Record<string, string>;
  skipped: boolean;
}

export interface CourtMapResponse {
  video_id: string;
  config_key: string;
  n_frames_mapped: number;
  skipped: boolean;
}

export interface RenderResponse {
  video_id: string;
  config_key: string;
  skipped: boolean;
}

export interface StageStatusResponse {
  status: string;
  config_key?: string;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  error?: string;
}

export interface PipelineStatusResponse {
  video_id: string;
  stages: Record<string, StageStatusResponse>;
}

// Settings
export interface TeamInfo {
  name: string;
  color: string;
}

export interface GameContext {
  teams: Record<string, TeamInfo>;
  roster: Record<string, string>; // jersey# -> player name
}

export interface AdvancedSettings {
  confidence: number;
  iou_threshold: number;
  ocr_interval: number;
  crop_scale: number;
  stride: number;
}

export interface AnalysisSettings {
  game_context: GameContext;
  advanced: AdvancedSettings;
}

// Chat
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
```

- [ ] **Step 2: Rewrite api.ts**

```typescript
// src/lib/api.ts

const API_BASE = "";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new ApiError(await res.text(), res.status);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new ApiError(await res.text(), res.status);
  return res.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(await res.text(), res.status);
  return res.json();
}

// Video + STT
export const downloadVideo = (url: string) =>
  post<import("./types").DownloadResponse>("/api/download", { url });

export const transcribeVideo = (videoId: string, useYoutubeCaptions = true) =>
  post<import("./types").TranscribeResponse>(
    `/api/transcribe/${videoId}?use_youtube_captions=${useYoutubeCaptions}`
  );

// Settings
export const getSettings = (videoId: string) =>
  get<import("./types").AnalysisSettings>(`/api/settings/${videoId}`);

export const saveSettings = (videoId: string, settings: import("./types").AnalysisSettings) =>
  put<import("./types").AnalysisSettings>(`/api/settings/${videoId}`, settings);

// Vision pipeline
export const detectPlayers = (videoId: string, params?: object) =>
  post<import("./types").DetectResponse>(`/api/vision/detect/${videoId}`, params);

export const trackPlayers = (videoId: string, body: { det_config_key: string }) =>
  post<import("./types").TrackResponse>(`/api/vision/track/${videoId}`, body);

export const classifyTeams = (videoId: string, body: { det_config_key: string; stride?: number; crop_scale?: number }) =>
  post<import("./types").ClassifyTeamsResponse>(`/api/vision/classify-teams/${videoId}`, body);

export const ocrJerseys = (videoId: string, body: { track_config_key: string; ocr_interval?: number }) =>
  post<import("./types").OCRResponse>(`/api/vision/ocr/${videoId}`, body);

export const mapCourt = (videoId: string, body: { det_config_key: string }) =>
  post<import("./types").CourtMapResponse>(`/api/vision/court-map/${videoId}`, body);

export const renderVideo = (videoId: string, body: object) =>
  post<import("./types").RenderResponse>(`/api/vision/render/${videoId}`, body);

export const getPipelineStatus = (videoId: string) =>
  get<import("./types").PipelineStatusResponse>(`/api/vision/status/${videoId}`);

// Artifacts
export const getArtifact = (stage: string, videoId: string, configKey: string) =>
  get<unknown>(`/api/vision/artifacts/${stage}/${videoId}?config_key=${configKey}`);

// Captions
export const buildTimeline = (videoId: string) =>
  post<{ video_id: string; config_key: string; n_segments: number; source: string; skipped: boolean }>(
    `/api/captions/timeline/${videoId}`
  );

// URL helpers
export const getVideoUrl = (videoId: string) =>
  `/api/video/${videoId}/original`;
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts
git commit -m "feat: rewrite types and API client for vision pipeline"
```

---

### Task 3: Analysis settings context

**Files:**
- Create: `src/contexts/analysis-settings-context.tsx`

- [ ] **Step 1: Create analysis settings context**

```typescript
// src/contexts/analysis-settings-context.tsx
"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { AnalysisSettings, GameContext, AdvancedSettings } from "@/lib/types";
import { getSettings, saveSettings } from "@/lib/api";

const DEFAULT_SETTINGS: AnalysisSettings = {
  game_context: {
    teams: {
      "0": { name: "Team A", color: "#006BB6" },
      "1": { name: "Team B", color: "#007A33" },
    },
    roster: {},
  },
  advanced: {
    confidence: 0.4,
    iou_threshold: 0.9,
    ocr_interval: 5,
    crop_scale: 0.4,
    stride: 30,
  },
};

interface AnalysisSettingsContextValue {
  settings: AnalysisSettings;
  updateGameContext: (ctx: Partial<GameContext>) => void;
  updateAdvanced: (adv: Partial<AdvancedSettings>) => void;
  loadForVideo: (videoId: string) => Promise<void>;
}

const Ctx = createContext<AnalysisSettingsContextValue | null>(null);

export function AnalysisSettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_SETTINGS);
  const [videoId, setVideoId] = useState<string | null>(null);

  const persist = useCallback(
    (next: AnalysisSettings) => {
      setSettings(next);
      if (videoId) saveSettings(videoId, next).catch(console.error);
    },
    [videoId]
  );

  const updateGameContext = useCallback(
    (partial: Partial<GameContext>) => {
      persist({ ...settings, game_context: { ...settings.game_context, ...partial } });
    },
    [settings, persist]
  );

  const updateAdvanced = useCallback(
    (partial: Partial<AdvancedSettings>) => {
      persist({ ...settings, advanced: { ...settings.advanced, ...partial } });
    },
    [settings, persist]
  );

  const loadForVideo = useCallback(async (vid: string) => {
    setVideoId(vid);
    try {
      const saved = await getSettings(vid);
      setSettings(saved);
    } catch {
      setSettings(DEFAULT_SETTINGS);
    }
  }, []);

  return (
    <Ctx.Provider value={{ settings, updateGameContext, updateAdvanced, loadForVideo }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAnalysisSettings() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAnalysisSettings must be used within AnalysisSettingsProvider");
  return ctx;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/contexts/analysis-settings-context.tsx
git commit -m "feat: add analysis settings context with API persistence"
```

---

### Task 4: Pipeline hook

**Files:**
- Rewrite: `src/hooks/use-pipeline.ts`

- [ ] **Step 1: Rewrite pipeline hook**

```typescript
// src/hooks/use-pipeline.ts
"use client";

import { useCallback, useReducer } from "react";
import type { PipelineState, VisionStage, StageState, AnalysisSettings } from "@/lib/types";
import * as api from "@/lib/api";

const STAGES: VisionStage[] = [
  "download", "transcribe", "detect", "track", "ocr", "classify-teams", "court-map", "render",
];

const INITIAL_STAGE: StageState = { status: "pending" };

const INITIAL_STATE: PipelineState = {
  status: "idle",
  stages: Object.fromEntries(STAGES.map((s) => [s, INITIAL_STAGE])) as Record<VisionStage, StageState>,
  videoId: undefined,
};

type Action =
  | { type: "START"; videoId: string }
  | { type: "STAGE_ACTIVE"; stage: VisionStage }
  | { type: "STAGE_COMPLETE"; stage: VisionStage; result: unknown; config_key?: string; duration_ms: number }
  | { type: "STAGE_ERROR"; stage: VisionStage; error: string }
  | { type: "STAGE_SKIPPED"; stage: VisionStage; config_key?: string }
  | { type: "COMPLETE" }
  | { type: "RESET" };

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case "START":
      return { ...INITIAL_STATE, status: "running", videoId: action.videoId };
    case "STAGE_ACTIVE":
      return {
        ...state,
        stages: { ...state.stages, [action.stage]: { status: "active", started_at: Date.now() } },
      };
    case "STAGE_COMPLETE":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            status: "complete",
            result: action.result,
            config_key: action.config_key,
            duration_ms: action.duration_ms,
          },
        },
      };
    case "STAGE_SKIPPED":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: { status: "skipped", config_key: action.config_key },
        },
      };
    case "STAGE_ERROR":
      return {
        ...state,
        status: "error",
        stages: { ...state.stages, [action.stage]: { status: "error", error: action.error } },
      };
    case "COMPLETE":
      return { ...state, status: "complete" };
    case "RESET":
      return INITIAL_STATE;
    default:
      return state;
  }
}

async function runStage<T>(
  dispatch: React.Dispatch<Action>,
  stage: VisionStage,
  fn: () => Promise<T & { config_key?: string; skipped?: boolean }>,
): Promise<T & { config_key?: string; skipped?: boolean }> {
  dispatch({ type: "STAGE_ACTIVE", stage });
  const start = Date.now();
  try {
    const result = await fn();
    const duration_ms = Date.now() - start;
    if (result.skipped) {
      dispatch({ type: "STAGE_SKIPPED", stage, config_key: result.config_key });
    } else {
      dispatch({ type: "STAGE_COMPLETE", stage, result, config_key: result.config_key, duration_ms });
    }
    return result;
  } catch (err) {
    dispatch({ type: "STAGE_ERROR", stage, error: err instanceof Error ? err.message : String(err) });
    throw err;
  }
}

export function usePipeline() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  const runPipeline = useCallback(
    async (videoId: string, videoUrl: string, settings: AnalysisSettings) => {
      dispatch({ type: "START", videoId });

      try {
        // 1. Download
        await runStage(dispatch, "download", () => api.downloadVideo(videoUrl));

        // 2. Transcribe
        await runStage(dispatch, "transcribe", () => api.transcribeVideo(videoId));

        // 3. Detect
        const detectResult = await runStage(dispatch, "detect", () =>
          api.detectPlayers(videoId, {
            confidence: settings.advanced.confidence,
            iou_threshold: settings.advanced.iou_threshold,
          })
        );
        const detKey = detectResult.config_key!;

        // 4. Track
        const trackResult = await runStage(dispatch, "track", () =>
          api.trackPlayers(videoId, { det_config_key: detKey })
        );
        const trackKey = trackResult.config_key!;

        // 5. OCR — capture return value for render
        const ocrResult = await runStage(dispatch, "ocr", () =>
          api.ocrJerseys(videoId, {
            track_config_key: trackKey,
            ocr_interval: settings.advanced.ocr_interval,
          })
        );

        // 6. Classify teams
        const teamsResult = await runStage(dispatch, "classify-teams", () =>
          api.classifyTeams(videoId, {
            det_config_key: detKey,
            stride: settings.advanced.stride,
            crop_scale: settings.advanced.crop_scale,
          })
        );

        // 7. Court map — capture return value for render
        const courtResult = await runStage(dispatch, "court-map", () =>
          api.mapCourt(videoId, { det_config_key: detKey })
        );

        // 8. Render (stub — will get 501, catch and mark skipped)
        try {
          await runStage(dispatch, "render", () =>
            api.renderVideo(videoId, {
              det_config_key: detKey,
              track_config_key: trackKey,
              teams_config_key: teamsResult.config_key!,
              jerseys_config_key: ocrResult.config_key!,
              court_config_key: courtResult.config_key!,
            })
          );
        } catch {
          // Render is a 501 stub — mark as skipped, don't fail pipeline
          dispatch({ type: "STAGE_SKIPPED", stage: "render" });
        }

        dispatch({ type: "COMPLETE" });
      } catch {
        // Error already dispatched by runStage
      }
    },
    [] // no state dependencies — all config keys captured from return values
  );

  const reset = useCallback(() => dispatch({ type: "RESET" }), []);

  return { state, runPipeline, reset };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/use-pipeline.ts
git commit -m "feat: rewrite pipeline hook for 8-stage vision pipeline with config key chaining"
```

---

### Task 5: Analysis layout and tabs

**Files:**
- Create: `src/components/analysis-layout.tsx`
- Create: `src/components/analysis-tabs.tsx`
- Modify: `src/app/page.tsx`
- Modify: `src/app/layout.tsx`

- [ ] **Step 1: Create analysis-tabs.tsx**

A tab container using shadcn Tabs. Renders Pipeline, Players, Court, Chat tabs. Accepts `selectedTab` and `onTabChange` props, plus children slots for each tab content.

- [ ] **Step 2: Create analysis-layout.tsx**

Root client component. Props: `videos: Video[]`. State: `selectedVideoId`, `selectedTab`. Composes:
- `AppSidebar` (left)
- `VideoPlayer` (top)
- `AnalysisTabs` (bottom) containing `PipelineTable`, `PlayersTable`, `CourtView`, `ChatPanel`
- `PipelineStatusBar` (overlay)

Uses `usePipeline()` and `useAnalysisSettings()`. Calls `loadForVideo()` when video selection changes.

- [ ] **Step 3: Update page.tsx**

```typescript
// src/app/page.tsx
import { AnalysisLayout } from "@/components/analysis-layout";
import type { Video } from "@/lib/types";

const API_URL = process.env.API_URL || "http://localhost:8080";

export default async function Home() {
  const res = await fetch(`${API_URL}/api/videos`, { cache: "no-store" });
  const videos: Video[] = res.ok ? await res.json() : [];
  return <AnalysisLayout videos={videos} />;
}
```

- [ ] **Step 4: Update layout.tsx**

Replace `StudioSettingsProvider` with `AnalysisSettingsProvider`. Update metadata title to "BasketTube" and description.

- [ ] **Step 5: Delete replaced files**

Now that replacements are in place, delete the old files:

```bash
rm frontend/src/components/studio-layout.tsx
rm frontend/src/contexts/studio-settings-context.tsx
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add analysis layout with tabbed panels, delete old studio layout"
```

---

### Task 6: App sidebar rebrand

**Files:**
- Rewrite: `src/components/app-sidebar.tsx`

- [ ] **Step 1: Rewrite app-sidebar.tsx**

- Header: "BasketTube" with basketball icon instead of "Foreign Whispers"
- Video list with selection state
- "Analyze" button (disabled when pipeline is running)
- Settings gear button opens settings dialog
- Footer: "aegean.ai" branding

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/app-sidebar.tsx
git commit -m "feat: rebrand sidebar to BasketTube with Analyze button"
```

---

### Task 7: Pipeline table and status components

**Files:**
- Rewrite: `src/components/pipeline-table.tsx`
- Rewrite: `src/components/pipeline-cards.tsx`
- Rewrite: `src/components/pipeline-status-bar.tsx`

- [ ] **Step 1: Rewrite pipeline-table.tsx**

8 rows: Download, Transcribe, Detect, Track, OCR, Classify Teams, Court Map, Render. Columns: stage icon/name, description, status badge, duration, re-run button.

Re-run behavior: clicking "Re-run" on a stage opens the settings dialog so the user can adjust parameters, then clicking "Analyze" again re-runs the full pipeline with new settings. Since changed parameters produce new config keys, the API processes stages fresh where configs differ and skips stages where configs match. This is simpler than per-stage re-run and avoids needing a separate `rerunFrom()` hook function — the existing `runPipeline()` handles it naturally.

- [ ] **Step 2: Rewrite pipeline-cards.tsx**

4 metric cards: Players Detected (n_detections), Players Tracked (n_tracks), Teams Classified (palette keys), Court Frames Mapped (n_frames_mapped). Values from pipeline stage results.

- [ ] **Step 3: Rewrite pipeline-status-bar.tsx**

Shows active stage name + elapsed time. Uses `useElapsed()` hook. Stage names: "Detecting players...", "Tracking...", "Classifying teams...", etc.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pipeline-table.tsx frontend/src/components/pipeline-cards.tsx frontend/src/components/pipeline-status-bar.tsx
git commit -m "feat: rewrite pipeline table, cards, and status bar for vision stages"
```

---

### Task 8: Video canvas rewrite

**Files:**
- Rewrite: `src/components/video-canvas.tsx`

- [ ] **Step 1: Rewrite video-canvas.tsx**

Remove all variant/subtitle logic. Show source video only using the existing `VideoPlayer` component. Video source URL from `getVideoUrl(videoId)`. No subtitle tracks, no variant selector.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/video-canvas.tsx
git commit -m "feat: simplify video canvas — source video only, no variants"
```

---

### Task 9: Players table

**Files:**
- Create: `src/components/players-table.tsx`

- [ ] **Step 1: Create players-table.tsx**

Props: `videoId`, `pipelineState` (to read config keys), `roster` (from settings).

Behavior:
- If OCR/teams/tracks stages not complete → show "Run pipeline to detect players"
- Once complete, fetch artifacts via `getArtifact("jerseys", ...)`, `getArtifact("teams", ...)`, `getArtifact("tracks", ...)`
- Join: for each tracker_id in OCR players map, find team from assignments, find first frame from tracks
- Render table: jersey#, player name (from roster lookup), team name + color badge, first seen timestamp
- Click row → callback to seek video player

Uses `useEffect` to fetch artifacts when config keys change.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/players-table.tsx
git commit -m "feat: add players table with team colors and click-to-seek"
```

---

### Task 10: Court view and chat panel placeholders

**Files:**
- Create: `src/components/court-view.tsx`
- Create: `src/components/chat-panel.tsx`

- [ ] **Step 1: Create court-view.tsx**

Static placeholder: centered message "Court visualization coming soon" with a basketball court SVG or emoji background. Accepts `videoId` prop for future wiring.

- [ ] **Step 2: Create chat-panel.tsx**

Functional chat UI shell:
- `messages: ChatMessage[]` state (local, not persisted)
- ScrollArea showing messages
- Input box + send button at bottom
- On send: add user message to list, add placeholder assistant message "Chat backend coming soon — this will use AI to answer questions about the game."
- Uses shadcn Card, ScrollArea, Input, Button

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/court-view.tsx frontend/src/components/chat-panel.tsx
git commit -m "feat: add court view placeholder and chat panel shell"
```

---

### Task 11: Settings dialog rewrite

**Files:**
- Rewrite: `src/components/settings-dialog.tsx`

- [ ] **Step 1: Rewrite settings-dialog.tsx**

Two sections using shadcn Accordion or Tabs:

**Game Context** (always visible):
- Team 0: name input + color input
- Team 1: name input + color input
- Roster: table with jersey# → player name rows, add/remove buttons

**Advanced** (collapsible):
- Detection Confidence: slider (0-1, step 0.05)
- IOU Threshold: slider (0-1, step 0.05)
- OCR Interval: number input
- Team Crop Scale: slider (0-1, step 0.05)
- Sampling Stride: number input

All values read from / write to `useAnalysisSettings()` context.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/settings-dialog.tsx
git commit -m "feat: rewrite settings dialog with game context and advanced params"
```

---

### Task 12: Integration and smoke test

- [ ] **Step 1: Verify the app builds**

```bash
cd frontend && pnpm install && pnpm build
```

Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 2: Fix any import errors or missing references**

Common issues:
- Old imports of `StudioLayout`, `StudioSettingsProvider`, `config-id`
- Missing shadcn components (install via `pnpx shadcn@latest add <component>`)

- [ ] **Step 3: Verify dev server starts**

```bash
cd frontend && pnpm dev
```

Open http://localhost:3000 — should show BasketTube layout with sidebar, video area, and tabs.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes for frontend redesign"
```
