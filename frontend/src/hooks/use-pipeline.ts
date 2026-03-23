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
        // 1. Download (send video ID for local files, URL for YouTube)
        const downloadUrl = videoUrl === "local" ? videoId : videoUrl;
        await runStage(dispatch, "download", () => api.downloadVideo(downloadUrl));

        // 2. Transcribe
        await runStage(dispatch, "transcribe", () => api.transcribeVideo(videoId));

        // 3. Detect
        const detectResult = await runStage(dispatch, "detect", () =>
          api.detectPlayers(videoId, {
            confidence: settings.stages.detect.confidence,
            iou_threshold: settings.stages.detect.iou_threshold,
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
            ocr_interval: settings.stages.ocr.ocr_interval,
          })
        );

        // 6. Classify teams
        const teamsResult = await runStage(dispatch, "classify-teams", () =>
          api.classifyTeams(videoId, {
            det_config_key: detKey,
            stride: settings.stages.teams.stride,
            crop_scale: settings.stages.teams.crop_scale,
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
