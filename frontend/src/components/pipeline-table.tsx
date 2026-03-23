"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  PlayIcon,
  RotateCcwIcon,
} from "lucide-react";
import { useElapsed } from "@/hooks/use-elapsed";
import type { VisionStage, PipelineState, StageState } from "@/lib/types";
import { isStageReady, getDownstream } from "@/lib/stage-deps";

const STAGES: {
  key: VisionStage;
  label: string;
  icon: React.ElementType;
  description: string;
}[] = [
  { key: "download", label: "Download", icon: DownloadIcon, description: "Fetch video from the registry" },
  { key: "transcribe", label: "Transcribe", icon: MicIcon, description: "Extract audio commentary via STT" },
  { key: "detect", label: "Detect", icon: ScanSearchIcon, description: "Player detection with RF-DETR" },
  { key: "track", label: "Track", icon: RouteIcon, description: "Multi-object tracking with SAM2" },
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
  allStages,
  onRunStage,
  onRerunStage,
}: {
  stageKey: VisionStage;
  label: string;
  icon: React.ElementType;
  description: string;
  stage: StageState;
  allStages: PipelineState["stages"];
  onRunStage?: (stage: VisionStage) => void;
  onRerunStage?: (stage: VisionStage) => void;
}) {
  const elapsed = useElapsed(stage.status === "active" ? stage.started_at : undefined);
  const duration = stage.status === "active" ? elapsed : stage.duration_ms;

  const ready = isStageReady(stageKey, allStages);

  let actionButton: React.ReactNode = null;
  if (stage.status === "pending") {
    if (ready) {
      actionButton = (
        <Button variant="outline" size="sm" onClick={() => onRunStage?.(stageKey)}>
          <PlayIcon className="size-3.5 mr-1" />
          Run
        </Button>
      );
    } else {
      actionButton = (
        <Button variant="outline" size="sm" disabled title="Dependencies not yet complete">
          <PlayIcon className="size-3.5 mr-1" />
          Run
        </Button>
      );
    }
  } else if (stage.status === "active") {
    actionButton = null;
  } else if (stage.status === "complete") {
    actionButton = (
      <Button variant="ghost" size="sm" onClick={() => onRerunStage?.(stageKey)}>
        <RotateCcwIcon className="size-3.5 mr-1" />
        Re-run
      </Button>
    );
  } else if (stage.status === "skipped") {
    actionButton = (
      <Button variant="outline" size="sm" onClick={() => onRunStage?.(stageKey)}>
        <PlayIcon className="size-3.5 mr-1" />
        Run
      </Button>
    );
  } else if (stage.status === "error") {
    actionButton = (
      <Button variant="outline" size="sm" className="text-destructive" onClick={() => onRerunStage?.(stageKey)}>
        <RotateCcwIcon className="size-3.5 mr-1" />
        Re-run
      </Button>
    );
  }

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
      <TableCell className="text-right tabular-nums">
        {stage.status === "active" && stage.progress != null ? (
          <div className="flex items-center gap-2 justify-end">
            <div className="h-1.5 w-16 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${Math.round(stage.progress * 100)}%` }}
              />
            </div>
            <span className="text-xs tabular-nums">{Math.round(stage.progress * 100)}%</span>
          </div>
        ) : (
          formatDuration(duration)
        )}
      </TableCell>
      <TableCell className="w-[100px]">{actionButton}</TableCell>
    </TableRow>
  );
}

interface PipelineTableProps {
  state: PipelineState;
  onRunStage?: (stage: VisionStage) => void;
  onRerunStage?: (stage: VisionStage) => void;
}

export function PipelineTable({ state, onRunStage, onRerunStage }: PipelineTableProps) {
  return (
    <div className="px-4 lg:px-6">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[180px]">Stage</TableHead>
            <TableHead>Description</TableHead>
            <TableHead className="w-[110px]">Status</TableHead>
            <TableHead className="w-[100px] text-right">Duration</TableHead>
            <TableHead className="w-[100px]">Action</TableHead>
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
              allStages={state.stages}
              onRunStage={onRunStage}
              onRerunStage={onRerunStage}
            />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
