import type { VisionStage, StageState } from "./types";

export const STAGE_DEPS: Record<VisionStage, VisionStage[]> = {
  download: [],
  transcribe: ["download"],
  detect: ["download"],
  track: ["detect"],
  ocr: ["track"],
  "classify-teams": ["detect"],
  "court-map": ["detect"],
  render: ["track", "ocr", "classify-teams", "court-map"],
};

/** Get all downstream stages (transitive) */
export function getDownstream(stage: VisionStage): VisionStage[] {
  const result: VisionStage[] = [];
  const queue = [stage];
  const visited = new Set<VisionStage>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [s, deps] of Object.entries(STAGE_DEPS) as [VisionStage, VisionStage[]][]) {
      if (deps.includes(current) && !visited.has(s)) {
        visited.add(s);
        result.push(s);
        queue.push(s);
      }
    }
  }
  return result;
}

/** Check if a stage is ready (all deps complete/skipped, stage is pending) */
export function isStageReady(
  stage: VisionStage,
  stages: Record<VisionStage, StageState>,
): boolean {
  const deps = STAGE_DEPS[stage];
  if (!deps.length) return stages[stage].status === "pending";
  return (
    deps.every((d) => stages[d].status === "complete" || stages[d].status === "skipped") &&
    stages[stage].status === "pending"
  );
}
