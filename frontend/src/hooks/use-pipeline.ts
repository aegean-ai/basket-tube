"use client";

import { useCallback, useReducer } from "react";
import type { PipelineState, VisionStage, StageState, AnalysisSettings } from "@/lib/types";
import type { SSEEvent } from "@/hooks/use-sse";
import { useSSE } from "@/hooks/use-sse";
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
  | { type: "STAGE_STARTED"; stage: VisionStage; timestamp: number }
  | { type: "STAGE_PROGRESS"; stage: VisionStage; progress: number; frame: number; total_frames: number }
  | { type: "STAGE_COMPLETE"; stage: VisionStage; config_key?: string; duration_s: number }
  | { type: "STAGE_SKIPPED"; stage: VisionStage; config_key?: string }
  | { type: "STAGE_ERROR"; stage: VisionStage; error: string }
  | { type: "PIPELINE_STATE"; stages: Record<string, unknown> }
  | { type: "COMPLETE" }
  | { type: "RESET" };

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case "START":
      return { ...INITIAL_STATE, status: "running", videoId: action.videoId };
    case "STAGE_STARTED":
      return {
        ...state,
        status: "running",
        stages: {
          ...state.stages,
          [action.stage]: { status: "active", started_at: action.timestamp * 1000 },
        },
      };
    case "STAGE_PROGRESS":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            ...state.stages[action.stage as VisionStage],
            progress: action.progress,
            frame: action.frame,
            total_frames: action.total_frames,
          },
        },
      };
    case "STAGE_COMPLETE":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            status: "complete",
            config_key: action.config_key,
            duration_ms: action.duration_s * 1000,
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
        stages: {
          ...state.stages,
          [action.stage]: { status: "error", error: action.error },
        },
      };
    case "PIPELINE_STATE": {
      const hydrated = { ...state, status: "running" as const };
      if (action.stages) {
        for (const [key, val] of Object.entries(action.stages)) {
          if (key in hydrated.stages) {
            hydrated.stages = { ...hydrated.stages, [key]: { ...hydrated.stages[key as VisionStage], ...(val as object) } };
          }
        }
      }
      return hydrated;
    }
    case "COMPLETE":
      return { ...state, status: "complete" };
    case "RESET":
      return INITIAL_STATE;
    default:
      return state;
  }
}

export function usePipeline() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  const handleSSE = useCallback((event: SSEEvent) => {
    switch (event.event) {
      case "stage_started":
        dispatch({ type: "STAGE_STARTED", stage: event.stage as VisionStage, timestamp: event.timestamp ?? Date.now() / 1000 });
        break;
      case "stage_progress":
        dispatch({
          type: "STAGE_PROGRESS",
          stage: event.stage as VisionStage,
          progress: event.progress ?? 0,
          frame: event.frame ?? 0,
          total_frames: event.total_frames ?? 0,
        });
        break;
      case "stage_completed":
        dispatch({ type: "STAGE_COMPLETE", stage: event.stage as VisionStage, config_key: event.config_key, duration_s: event.duration_s ?? 0 });
        break;
      case "stage_skipped":
        dispatch({ type: "STAGE_SKIPPED", stage: event.stage as VisionStage, config_key: event.config_key });
        break;
      case "stage_error":
        dispatch({ type: "STAGE_ERROR", stage: event.stage as VisionStage, error: event.error ?? "Unknown error" });
        break;
      case "pipeline_state":
        dispatch({ type: "PIPELINE_STATE", stages: event.stages ?? {} });
        break;
      case "pipeline_completed":
        dispatch({ type: "COMPLETE" });
        break;
    }
  }, []);

  const { connected } = useSSE(state.videoId, { onEvent: handleSSE });

  const runPipeline = useCallback(
    async (videoId: string, _videoUrl: string, settings: AnalysisSettings) => {
      dispatch({ type: "START", videoId });
      try {
        await api.runFullPipeline(videoId, settings);
      } catch (err) {
        dispatch({ type: "STAGE_ERROR", stage: "download", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [],
  );

  const runStage = useCallback(
    async (stage: VisionStage, videoId: string, params: object) => {
      const stageEndpoints: Record<string, (id: string, p: object) => Promise<unknown>> = {
        detect: api.detectPlayers,
        track: api.trackPlayers,
        ocr: api.ocrJerseys,
        "classify-teams": api.classifyTeams,
        "court-map": api.mapCourt,
      };
      const fn = stageEndpoints[stage];
      if (fn) await fn(videoId, params);
    },
    [],
  );

  const rerunStage = useCallback(
    async (stage: VisionStage, videoId: string, configKey: string, params: object) => {
      await api.deleteArtifact(stage, videoId, configKey);
      await runStage(stage, videoId, params);
    },
    [runStage],
  );

  const cancelPipeline = useCallback(async () => {
    if (state.videoId) await api.cancelPipeline(state.videoId);
  }, [state.videoId]);

  const reset = useCallback(() => dispatch({ type: "RESET" }), []);

  return { state, runPipeline, runStage, rerunStage, cancelPipeline, reset, connected };
}
