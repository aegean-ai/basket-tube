"use client";

import { useElapsed } from "@/hooks/use-elapsed";
import type { PipelineState, VisionStage } from "@/lib/types";

const STAGE_DISPLAY_NAMES: Record<VisionStage, string> = {
  download: "Downloading...",
  transcribe: "Transcribing...",
  detect: "Detecting players...",
  track: "Tracking players...",
  ocr: "Reading jerseys...",
  "classify-teams": "Classifying teams...",
  "court-map": "Mapping court...",
  render: "Rendering...",
};

const STAGE_ORDER: VisionStage[] = [
  "download",
  "transcribe",
  "detect",
  "track",
  "ocr",
  "classify-teams",
  "court-map",
  "render",
];

function formatElapsed(ms: number | undefined): string {
  if (ms == null) return "";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

interface PipelineStatusBarProps {
  state: PipelineState;
}

export function PipelineStatusBar({ state }: PipelineStatusBarProps) {
  const activeStage = STAGE_ORDER.find(
    (key) => state.stages[key].status === "active"
  );
  const startedAt = activeStage ? state.stages[activeStage].started_at : undefined;
  const elapsed = useElapsed(startedAt);

  if (state.status === "idle") return null;

  let message: string;
  if (state.status === "complete") {
    message = "Analysis complete.";
  } else if (state.status === "error") {
    const errorStage = STAGE_ORDER.find(
      (key) => state.stages[key].status === "error"
    );
    const errorMsg = errorStage ? state.stages[errorStage].error : "Unknown error";
    message = `Error in ${errorStage ?? "pipeline"}: ${errorMsg}`;
  } else if (activeStage) {
    const elapsedStr = formatElapsed(elapsed);
    message = `${STAGE_DISPLAY_NAMES[activeStage]}${elapsedStr ? ` (${elapsedStr})` : ""}`;
  } else {
    message = "Starting analysis...";
  }

  return (
    <div className="border-t bg-muted/50 px-4 py-1.5 text-xs text-muted-foreground font-mono truncate lg:px-6">
      {message}
    </div>
  );
}
