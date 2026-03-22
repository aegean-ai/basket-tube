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
