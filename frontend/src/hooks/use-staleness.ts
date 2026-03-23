"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisSettings, StalenessMap } from "@/lib/types";
import { checkStaleness } from "@/lib/api";

/**
 * Calls the staleness endpoint whenever settings change (debounced 500ms).
 * Returns a map of stage → { stale, reason } so the pipeline table can
 * show "Outdated" badges.
 */
export function useStaleness(
  videoId: string | undefined,
  settings: AnalysisSettings,
) {
  const [staleness, setStaleness] = useState<StalenessMap>({});
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!videoId) {
      setStaleness({});
      return;
    }

    // Debounce 500ms
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const result = await checkStaleness(videoId, settings);
        setStaleness(result);
      } catch {
        // Endpoint may not be available yet — ignore
        setStaleness({});
      }
    }, 500);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [videoId, settings]);

  return staleness;
}
