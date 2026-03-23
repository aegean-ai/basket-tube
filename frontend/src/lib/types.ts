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

export type StageStatus = "pending" | "ready" | "active" | "complete" | "skipped" | "error";

export interface StageState {
  status: StageStatus;
  config_key?: string;
  result?: unknown;
  error?: string;
  duration_ms?: number;
  started_at?: number;
  progress?: number;
  frame?: number;
  total_frames?: number;
}

export interface PipelineState {
  status: "idle" | "running" | "complete" | "error";
  stages: Record<VisionStage, StageState>;
  videoId?: string;
}

// UI state (TabId removed — analytics views are in sidebar now)

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

export interface TranscribeSettings {
  model: string;
  use_youtube_captions: boolean;
}

export interface DetectSettings {
  model_id: string;
  confidence: number;
  iou_threshold: number;
}

export interface TrackSettings {
  iou_threshold: number;
  track_activation_threshold: number;
  lost_track_buffer: number;
}

export interface OCRSettings {
  model_id: string;
  ocr_interval: number;
  n_consecutive: number;
}

export interface TeamsSettings {
  embedding_model: string;
  n_teams: number;
  crop_scale: number;
  stride: number;
}

export interface CourtMapSettings {
  model_id: string;
  keypoint_confidence: number;
  anchor_confidence: number;
}

export interface StageSettings {
  transcribe: TranscribeSettings;
  detect: DetectSettings;
  track: TrackSettings;
  ocr: OCRSettings;
  teams: TeamsSettings;
  court_map: CourtMapSettings;
}

export interface AnalysisSettings {
  game_context: GameContext;
  stages: StageSettings;
}

// Staleness
export interface StalenessEntry {
  stale: boolean;
  reason?: string;
}
export type StalenessMap = Record<string, StalenessEntry>;

// Chat
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
