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
