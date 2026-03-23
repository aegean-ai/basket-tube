"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface SSEEvent {
  event: string;
  stage?: string;
  config_key?: string;
  timestamp?: number;
  progress?: number;
  frame?: number;
  total_frames?: number;
  duration_s?: number;
  error?: string;
  stages_completed?: number;
  stages_skipped?: number;
  stages?: Record<string, unknown>;
}

interface UseSSEOptions {
  onEvent: (event: SSEEvent) => void;
}

export function useSSE(videoId: string | undefined, { onEvent }: UseSSEOptions) {
  const [connected, setConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (!videoId) return;

    const es = new EventSource(`/api/pipeline/events/${videoId}`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    const eventTypes = [
      "pipeline_state", "stage_started", "stage_progress",
      "stage_completed", "stage_skipped", "stage_error", "pipeline_completed",
    ];

    for (const type of eventTypes) {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data) as SSEEvent;
          data.event = type;
          onEventRef.current(data);
        } catch { /* ignore parse errors */ }
      });
    }

    return es;
  }, [videoId]);

  const disconnect = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setConnected(false);
  }, []);

  useEffect(() => {
    const es = connect();
    return () => {
      es?.close();
      setConnected(false);
    };
  }, [connect]);

  return { connected, disconnect };
}
