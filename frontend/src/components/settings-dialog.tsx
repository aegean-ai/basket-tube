"use client";

import { useState } from "react";
import { Dialog } from "@base-ui/react/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  XIcon,
  PlusIcon,
  Trash2Icon,
  UserIcon,
  DownloadIcon,
  MicIcon,
  ScanSearchIcon,
  RouteIcon,
  ScanTextIcon,
  UsersIcon,
  MapIcon,
} from "lucide-react";
import { useAnalysisSettings } from "@/contexts/analysis-settings-context";

type SettingsSection =
  | "game_context"
  | "download"
  | "transcribe"
  | "detect"
  | "track"
  | "ocr"
  | "teams"
  | "court_map";

const SECTIONS: { key: SettingsSection; label: string; icon: React.ElementType }[] = [
  { key: "game_context", label: "Game Context", icon: UserIcon },
  { key: "download", label: "Download", icon: DownloadIcon },
  { key: "transcribe", label: "Transcribe", icon: MicIcon },
  { key: "detect", label: "Detect", icon: ScanSearchIcon },
  { key: "track", label: "Track", icon: RouteIcon },
  { key: "ocr", label: "OCR", icon: ScanTextIcon },
  { key: "teams", label: "Teams", icon: UsersIcon },
  { key: "court_map", label: "Court Map", icon: MapIcon },
];

/* ---------- Section components ---------- */

function GameContextSettings() {
  const { settings, updateGameContext } = useAnalysisSettings();
  const { game_context } = settings;

  const setTeamName = (teamId: string, name: string) => {
    updateGameContext({
      teams: {
        ...game_context.teams,
        [teamId]: { ...game_context.teams[teamId], name },
      },
    });
  };

  const setTeamColor = (teamId: string, color: string) => {
    updateGameContext({
      teams: {
        ...game_context.teams,
        [teamId]: { ...game_context.teams[teamId], color },
      },
    });
  };

  const rosterEntries = Object.entries(game_context.roster);

  const addPlayer = () => {
    let n = rosterEntries.length + 1;
    while (game_context.roster[String(n)] !== undefined) n++;
    updateGameContext({
      roster: { ...game_context.roster, [String(n)]: "" },
    });
  };

  const updatePlayerJersey = (oldJersey: string, newJersey: string) => {
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(game_context.roster)) {
      next[k === oldJersey ? newJersey : k] = v;
    }
    updateGameContext({ roster: next });
  };

  const updatePlayerName = (jersey: string, name: string) => {
    updateGameContext({
      roster: { ...game_context.roster, [jersey]: name },
    });
  };

  const removePlayer = (jersey: string) => {
    const next = { ...game_context.roster };
    delete next[jersey];
    updateGameContext({ roster: next });
  };

  return (
    <div className="space-y-6">
      {/* Teams */}
      <div className="space-y-3">
        {(["0", "1"] as const).map((teamId, idx) => {
          const team = game_context.teams[teamId] ?? {
            name: `Team ${String.fromCharCode(65 + idx)}`,
            color: "#888888",
          };
          return (
            <div key={teamId} className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground w-14 shrink-0">
                Team {idx}
              </span>
              <Input
                value={team.name}
                onChange={(e) => setTeamName(teamId, e.target.value)}
                placeholder={`Team ${idx} name`}
                className="flex-1"
              />
              <input
                type="color"
                value={team.color}
                onChange={(e) => setTeamColor(teamId, e.target.value)}
                className="h-8 w-10 cursor-pointer rounded-md border border-input bg-transparent p-0.5"
                title="Team color"
              />
            </div>
          );
        })}
      </div>

      <Separator />

      {/* Roster */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">Roster</p>
          <Button variant="outline" size="sm" onClick={addPlayer} className="gap-1.5">
            <PlusIcon className="size-3.5" />
            Add Player
          </Button>
        </div>

        {rosterEntries.length === 0 ? (
          <p className="text-xs text-muted-foreground py-2">
            No players added yet. Click &quot;Add Player&quot; to start.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-24">Jersey #</TableHead>
                <TableHead>Player Name</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rosterEntries.map(([jersey, name]) => (
                <TableRow key={jersey}>
                  <TableCell>
                    <Input
                      value={jersey}
                      onChange={(e) => updatePlayerJersey(jersey, e.target.value)}
                      className="w-20 font-mono"
                    />
                  </TableCell>
                  <TableCell>
                    <Input
                      value={name}
                      onChange={(e) => updatePlayerName(jersey, e.target.value)}
                      placeholder="Player name"
                    />
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => removePlayer(jersey)}
                      aria-label={`Remove jersey ${jersey}`}
                    >
                      <Trash2Icon className="size-3.5 text-destructive" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}

function DownloadSettings() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Video source</p>
          <p className="text-xs text-muted-foreground">yt-dlp downloads the best available quality.</p>
        </div>
        <span className="bg-muted px-2 py-1 rounded text-xs text-muted-foreground">Best available</span>
      </div>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Auto-fetch captions</p>
          <p className="text-xs text-muted-foreground">Download YouTube closed captions when available.</p>
        </div>
        <span className="bg-muted px-2 py-1 rounded text-xs text-muted-foreground">Always on</span>
      </div>
    </div>
  );
}

function TranscribeSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.transcribe;

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">STT Model</label>
        <Input
          value={cfg.model}
          onChange={(e) => updateStage("transcribe", { model: e.target.value })}
          className="font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground">Whisper model used for transcription.</p>
      </div>
      <Separator />
      <div className="flex items-start gap-3">
        <Checkbox
          id="use-yt-captions"
          checked={cfg.use_youtube_captions}
          onCheckedChange={(checked) =>
            updateStage("transcribe", { use_youtube_captions: checked === true })
          }
          className="mt-0.5"
        />
        <div>
          <Label htmlFor="use-yt-captions" className="text-sm font-medium cursor-pointer">
            Use YouTube Captions
          </Label>
          <p className="text-xs text-muted-foreground mt-1">
            When available, use YouTube&apos;s closed captions instead of running Whisper.
            Uncheck to always run Whisper STT.
          </p>
        </div>
      </div>
    </div>
  );
}

function DetectSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.detect;

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Model ID</label>
        <Input
          value={cfg.model_id}
          onChange={(e) => updateStage("detect", { model_id: e.target.value })}
          className="font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground">Roboflow model for player detection.</p>
      </div>
      <Separator />
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">Detection Confidence</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.confidence.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.confidence}
          onChange={(e) => updateStage("detect", { confidence: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">IOU Threshold</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.iou_threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.iou_threshold}
          onChange={(e) => updateStage("detect", { iou_threshold: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
    </div>
  );
}

function TrackSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.track;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Tracker</p>
          <p className="text-xs text-muted-foreground">Multi-object tracker algorithm.</p>
        </div>
        <span className="bg-muted px-2 py-1 rounded text-xs text-muted-foreground">ByteTrack</span>
      </div>
      <Separator />
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">IOU Threshold</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.iou_threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.iou_threshold}
          onChange={(e) => updateStage("track", { iou_threshold: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">Activation Threshold</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.track_activation_threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.track_activation_threshold}
          onChange={(e) => updateStage("track", { track_activation_threshold: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
      <div className="flex items-center justify-between gap-4">
        <label className="text-sm">Lost Track Buffer (frames)</label>
        <Input
          type="number"
          min={1}
          value={cfg.lost_track_buffer}
          onChange={(e) => updateStage("track", { lost_track_buffer: Number(e.target.value) })}
          className="w-24 text-right font-mono"
        />
      </div>
    </div>
  );
}

function OCRSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.ocr;

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Model ID</label>
        <Input
          value={cfg.model_id}
          onChange={(e) => updateStage("ocr", { model_id: e.target.value })}
          className="font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground">Roboflow model for jersey number OCR.</p>
      </div>
      <Separator />
      <div className="flex items-center justify-between gap-4">
        <label className="text-sm">OCR Interval (frames)</label>
        <Input
          type="number"
          min={1}
          value={cfg.ocr_interval}
          onChange={(e) => updateStage("ocr", { ocr_interval: Number(e.target.value) })}
          className="w-24 text-right font-mono"
        />
      </div>
      <div className="flex items-center justify-between gap-4">
        <label className="text-sm">Consecutive Reads</label>
        <Input
          type="number"
          min={1}
          value={cfg.n_consecutive}
          onChange={(e) => updateStage("ocr", { n_consecutive: Number(e.target.value) })}
          className="w-24 text-right font-mono"
        />
      </div>
    </div>
  );
}

function TeamsSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.teams;

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Embedding Model</label>
        <Input
          value={cfg.embedding_model}
          onChange={(e) => updateStage("teams", { embedding_model: e.target.value })}
          className="font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground">SigLIP model for jersey color embeddings.</p>
      </div>
      <Separator />
      <div className="flex items-center justify-between gap-4">
        <label className="text-sm">Number of Teams</label>
        <Input
          type="number"
          min={2}
          max={4}
          value={cfg.n_teams}
          onChange={(e) => updateStage("teams", { n_teams: Number(e.target.value) })}
          className="w-24 text-right font-mono"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">Crop Scale</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.crop_scale.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.crop_scale}
          onChange={(e) => updateStage("teams", { crop_scale: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
      <div className="flex items-center justify-between gap-4">
        <label className="text-sm">Sampling Stride (frames)</label>
        <Input
          type="number"
          min={1}
          value={cfg.stride}
          onChange={(e) => updateStage("teams", { stride: Number(e.target.value) })}
          className="w-24 text-right font-mono"
        />
      </div>
    </div>
  );
}

function CourtMapSettings() {
  const { settings, updateStage } = useAnalysisSettings();
  const cfg = settings.stages.court_map;

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Model ID</label>
        <Input
          value={cfg.model_id}
          onChange={(e) => updateStage("court_map", { model_id: e.target.value })}
          className="font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground">Roboflow model for court keypoint detection.</p>
      </div>
      <Separator />
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">Keypoint Confidence</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.keypoint_confidence.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.keypoint_confidence}
          onChange={(e) => updateStage("court_map", { keypoint_confidence: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm">Anchor Confidence</label>
          <span className="text-xs text-muted-foreground font-mono">{cfg.anchor_confidence.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={cfg.anchor_confidence}
          onChange={(e) => updateStage("court_map", { anchor_confidence: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>
    </div>
  );
}

/* ---------- Section metadata ---------- */

const SECTION_CONTENT: Record<SettingsSection, { title: string; description: string; component: React.FC }> = {
  game_context: { title: "Game Context", description: "Team names, colors, and player roster.", component: GameContextSettings },
  download: { title: "Download", description: "Video download and caption fetching.", component: DownloadSettings },
  transcribe: { title: "Transcribe", description: "Speech-to-text via Whisper.", component: TranscribeSettings },
  detect: { title: "Detect", description: "Player detection via RF-DETR.", component: DetectSettings },
  track: { title: "Track", description: "Multi-object tracking via ByteTrack.", component: TrackSettings },
  ocr: { title: "OCR", description: "Jersey number recognition.", component: OCRSettings },
  teams: { title: "Teams", description: "Team classification via color embeddings.", component: TeamsSettings },
  court_map: { title: "Court Map", description: "Court keypoint detection and homography.", component: CourtMapSettings },
};

/* ---------- Main dialog ---------- */

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const [activeSection, setActiveSection] = useState<SettingsSection>("game_context");
  const { title, description, component: Content } = SECTION_CONTENT[activeSection];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/50 transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0 supports-backdrop-filter:backdrop-blur-sm" />
        <Dialog.Popup className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-background shadow-2xl transition duration-200 data-ending-style:opacity-0 data-ending-style:scale-95 data-starting-style:opacity-0 data-starting-style:scale-95 w-[720px] max-h-[80vh]">
          <div className="flex h-[560px]">
            {/* Sidebar nav */}
            <nav className="w-48 border-r p-4 flex flex-col gap-1 shrink-0">
              <Dialog.Title className="text-sm font-semibold mb-3 px-2">Settings</Dialog.Title>
              {SECTIONS.map(({ key, label, icon: Icon }) => (
                <button
                  type="button"
                  key={key}
                  onClick={() => setActiveSection(key)}
                  className={`flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors ${
                    activeSection === key
                      ? "bg-accent text-accent-foreground font-medium"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                  }`}
                >
                  <Icon className="size-3.5" />
                  {label}
                </button>
              ))}
            </nav>

            {/* Content area */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="mb-1">
                <h3 className="text-lg font-medium">{title}</h3>
                <Dialog.Description className="text-sm text-muted-foreground">{description}</Dialog.Description>
              </div>
              <Separator className="my-4" />
              <Content />
            </div>
          </div>

          {/* Close button */}
          <Dialog.Close
            render={<Button variant="ghost" size="icon-sm" className="absolute top-3 right-3" />}
          >
            <XIcon />
            <span className="sr-only">Close</span>
          </Dialog.Close>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
