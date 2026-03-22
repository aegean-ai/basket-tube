"use client";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DownloadIcon,
  MicIcon,
  ScanSearchIcon,
  RouteIcon,
  ScanTextIcon,
  UsersIcon,
  MapIcon,
  VideoIcon,
} from "lucide-react";
import { useElapsed } from "@/hooks/use-elapsed";
import type { VisionStage, PipelineState, StageState } from "@/lib/types";

const STAGES: {
  key: VisionStage;
  label: string;
  icon: React.ElementType;
  description: string;
}[] = [
  { key: "download", label: "Download", icon: DownloadIcon, description: "Fetch video from the registry" },
  { key: "transcribe", label: "Transcribe", icon: MicIcon, description: "Extract audio commentary via STT" },
  { key: "detect", label: "Detect", icon: ScanSearchIcon, description: "Player detection with RF-DETR" },
  { key: "track", label: "Track", icon: RouteIcon, description: "Multi-object tracking with ByteTrack" },
  { key: "ocr", label: "OCR", icon: ScanTextIcon, description: "Jersey number recognition" },
  { key: "classify-teams", label: "Classify Teams", icon: UsersIcon, description: "Team classification via color clustering" },
  { key: "court-map", label: "Court Map", icon: MapIcon, description: "Homography-based bird's-eye mapping" },
  { key: "render", label: "Render", icon: VideoIcon, description: "Annotate and render output video" },
];

function statusBadge(status: string) {
  switch (status) {
    case "active":
      return <Badge variant="secondary" className="bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">Running</Badge>;
    case "complete":
      return <Badge variant="default" className="bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300">Done</Badge>;
    case "skipped":
      return <Badge variant="outline" className="text-muted-foreground">Skipped</Badge>;
    case "error":
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="outline" className="text-muted-foreground/50">Pending</Badge>;
  }
}

function formatDuration(ms: number | undefined): string {
  if (ms == null) return "--";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function StageRow({
  stageKey,
  label,
  icon: Icon,
  description,
  stage,
}: {
  stageKey: string;
  label: string;
  icon: React.ElementType;
  description: string;
  stage: StageState;
}) {
  const elapsed = useElapsed(stage.status === "active" ? stage.started_at : undefined);
  const duration = stage.status === "active" ? elapsed : stage.duration_ms;

  return (
    <TableRow>
      <TableCell className="font-medium">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-muted-foreground" />
          {label}
        </div>
      </TableCell>
      <TableCell className="text-muted-foreground">{description}</TableCell>
      <TableCell>{statusBadge(stage.status)}</TableCell>
      <TableCell className="text-right tabular-nums">{formatDuration(duration)}</TableCell>
    </TableRow>
  );
}

interface PipelineTableProps {
  state: PipelineState;
}

export function PipelineTable({ state }: PipelineTableProps) {
  return (
    <div className="px-4 lg:px-6">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[180px]">Stage</TableHead>
            <TableHead>Description</TableHead>
            <TableHead className="w-[110px]">Status</TableHead>
            <TableHead className="w-[100px] text-right">Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {STAGES.map(({ key, label, icon, description }) => (
            <StageRow
              key={key}
              stageKey={key}
              label={label}
              icon={icon}
              description={description}
              stage={state.stages[key]}
            />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
