"use client";

import { getVideoUrl } from "@/lib/api";
import { VideoPlayer } from "./video-player";

interface VideoCanvasProps {
  videoId?: string;
}

export function VideoCanvas({ videoId }: VideoCanvasProps) {
  if (!videoId) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Select a video
      </div>
    );
  }

  return (
    <div className="p-4">
      <VideoPlayer src={getVideoUrl(videoId)} title="Source Video" />
    </div>
  );
}
