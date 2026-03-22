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
