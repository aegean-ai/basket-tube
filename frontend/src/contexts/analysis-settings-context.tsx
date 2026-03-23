"use client";

import { createContext, useCallback, useContext, useState } from "react";
import type { AnalysisSettings, GameContext, StageSettings } from "@/lib/types";
import { getSettings, saveSettings } from "@/lib/api";

const DEFAULT_SETTINGS: AnalysisSettings = {
  game_context: {
    teams: {
      "0": { name: "Team A", color: "#006BB6" },
      "1": { name: "Team B", color: "#007A33" },
    },
    roster: {},
  },
  stages: {
    transcribe: { model: "Systran/faster-whisper-medium", use_youtube_captions: true },
    detect: { model_id: "basketball-player-detection-3-ycjdo/4", confidence: 0.4, iou_threshold: 0.9 },
    track: { iou_threshold: 0.5, track_activation_threshold: 0.25, lost_track_buffer: 30 },
    ocr: { model_id: "basketball-jersey-numbers-ocr/3", ocr_interval: 5, n_consecutive: 3 },
    teams: { embedding_model: "google/siglip-base-patch16-224", n_teams: 2, crop_scale: 0.4, stride: 30 },
    court_map: { model_id: "basketball-court-detection-2/14", keypoint_confidence: 0.3, anchor_confidence: 0.5 },
  },
};

interface AnalysisSettingsContextValue {
  settings: AnalysisSettings;
  updateGameContext: (ctx: Partial<GameContext>) => void;
  updateStage: <K extends keyof StageSettings>(stage: K, partial: Partial<StageSettings[K]>) => void;
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

  const updateStage = useCallback(
    <K extends keyof StageSettings>(stage: K, partial: Partial<StageSettings[K]>) => {
      persist({
        ...settings,
        stages: {
          ...settings.stages,
          [stage]: { ...settings.stages[stage], ...partial },
        },
      });
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
    <Ctx.Provider value={{ settings, updateGameContext, updateStage, loadForVideo }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAnalysisSettings() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAnalysisSettings must be used within AnalysisSettingsProvider");
  return ctx;
}
