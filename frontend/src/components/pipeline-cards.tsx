"use client";

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ScanSearchIcon,
  RouteIcon,
  UsersIcon,
  MapIcon,
} from "lucide-react";
import type {
  PipelineState,
  DetectResponse,
  TrackResponse,
  ClassifyTeamsResponse,
  CourtMapResponse,
} from "@/lib/types";

interface PipelineCardsProps {
  state: PipelineState;
}

export function PipelineCards({ state }: PipelineCardsProps) {
  const detectResult = state.stages.detect.result as DetectResponse | undefined;
  const trackResult = state.stages.track.result as TrackResponse | undefined;
  const classifyTeamsResult = state.stages["classify-teams"].result as ClassifyTeamsResponse | undefined;
  const courtMapResult = state.stages["court-map"].result as CourtMapResponse | undefined;

  const metrics: {
    label: string;
    value: string;
    icon: React.ElementType;
  }[] = [
    {
      label: "Detections",
      value: detectResult?.n_detections != null ? `${detectResult.n_detections}` : "--",
      icon: ScanSearchIcon,
    },
    {
      label: "Tracks",
      value: trackResult?.n_tracks != null ? `${trackResult.n_tracks}` : "--",
      icon: RouteIcon,
    },
    {
      label: "Teams",
      value: classifyTeamsResult?.palette != null
        ? `${Object.keys(classifyTeamsResult.palette).length}`
        : "--",
      icon: UsersIcon,
    },
    {
      label: "Court Frames",
      value: courtMapResult?.n_frames_mapped != null ? `${courtMapResult.n_frames_mapped}` : "--",
      icon: MapIcon,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 px-4 lg:px-6 @xl/main:grid-cols-4">
      {metrics.map(({ label, value, icon: Icon }) => (
        <Card key={label} className="@container/card">
          <CardHeader>
            <CardDescription className="flex items-center gap-1.5">
              <Icon className="size-3.5" />
              {label}
            </CardDescription>
            <CardTitle className="text-lg font-semibold tabular-nums">
              {value}
            </CardTitle>
          </CardHeader>
        </Card>
      ))}
    </div>
  );
}
