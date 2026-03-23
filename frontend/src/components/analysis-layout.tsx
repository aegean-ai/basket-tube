"use client";

import { useState, useEffect } from "react";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { AnalysisTabs } from "@/components/analysis-tabs";
import { PipelineStatusBar } from "@/components/pipeline-status-bar";
import { PipelineTable } from "@/components/pipeline-table";
import { PipelineCards } from "@/components/pipeline-cards";
import { PlayersTable } from "@/components/players-table";
import { CourtView } from "@/components/court-view";
import { ChatPanel } from "@/components/chat-panel";
import { VideoCanvas } from "@/components/video-canvas";
import { SettingsDialog } from "@/components/settings-dialog";
import { usePipeline } from "@/hooks/use-pipeline";
import { useAnalysisSettings } from "@/contexts/analysis-settings-context";
import { useStaleness } from "@/hooks/use-staleness";
import * as api from "@/lib/api";
import type { Video, TabId, VisionStage } from "@/lib/types";

interface AnalysisLayoutProps {
  videos: Video[];
}

export function AnalysisLayout({ videos }: AnalysisLayoutProps) {
  const [selectedVideoId, setSelectedVideoId] = useState<string | undefined>(
    videos[0]?.id
  );
  const [selectedTab, setSelectedTab] = useState<TabId>("pipeline");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { state, runPipeline, cancelPipeline, markStageActive, markStageComplete, markStageError, reset, connected } = usePipeline();
  const { settings, loadForVideo } = useAnalysisSettings();
  const staleness = useStaleness(selectedVideoId, settings);

  const selectedVideo = videos.find((v) => v.id === selectedVideoId);

  useEffect(() => {
    if (selectedVideoId) {
      loadForVideo(selectedVideoId);
    }
  }, [selectedVideoId, loadForVideo]);

  const handleAnalyze = () => {
    if (!selectedVideo) return;
    runPipeline(selectedVideo.id, selectedVideo.url, settings);
  };

  const handleVideoSelect = (id: string) => {
    if (state.status === "running") return;
    setSelectedVideoId(id);
    reset();
  };

  /**
   * Run a single stage directly via its API endpoint.
   * Does NOT reset other stages. Only updates this stage's status.
   */
  const handleRunStage = async (stage: VisionStage) => {
    if (!selectedVideo) return;
    const vid = selectedVideo.id;
    const s = settings.stages;
    const detKey = state.stages.detect.config_key ?? "";
    const trackKey = state.stages.track.config_key ?? "";

    markStageActive(stage);
    try {
      switch (stage) {
        case "download": {
          await api.downloadVideo(selectedVideo.url === "local" ? vid : selectedVideo.url);
          markStageComplete("download");
          break;
        }
        case "transcribe": {
          const res = await api.transcribeVideo(vid, s.transcribe.use_youtube_captions);
          markStageComplete("transcribe", { skipped: res.skipped });
          break;
        }
        case "detect": {
          const res = await api.detectPlayers(vid, { model_id: s.detect.model_id, confidence: s.detect.confidence, iou_threshold: s.detect.iou_threshold });
          markStageComplete("detect", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
        case "track": {
          const res = await api.trackPlayers(vid, { det_config_key: detKey });
          markStageComplete("track", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
        case "ocr": {
          const res = await api.ocrJerseys(vid, { track_config_key: trackKey, ocr_interval: s.ocr.ocr_interval });
          markStageComplete("ocr", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
        case "classify-teams": {
          const res = await api.classifyTeams(vid, { det_config_key: detKey, stride: s.teams.stride, crop_scale: s.teams.crop_scale });
          markStageComplete("classify-teams", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
        case "court-map": {
          const res = await api.mapCourt(vid, { det_config_key: detKey });
          markStageComplete("court-map", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
        case "render": {
          const teamsKey = state.stages["classify-teams"].config_key ?? "";
          const jerseysKey = state.stages.ocr.config_key ?? "";
          const res = await api.renderVideo(vid, {
            det_config_key: detKey,
            track_config_key: trackKey,
            teams_config_key: teamsKey,
            jerseys_config_key: jerseysKey,
          });
          markStageComplete("render", { skipped: res.skipped, config_key: res.config_key });
          break;
        }
      }
    } catch (err) {
      markStageError(stage, err instanceof Error ? err.message : String(err));
    }
  };

  /**
   * Re-run = delete artifact + run stage.
   * The API endpoint clears cache automatically when artifact is missing.
   */
  const handleRerunStage = async (stage: VisionStage) => {
    if (!selectedVideo) return;
    const configKey = state.stages[stage].config_key ?? "";
    // Map stage name to artifact directory name
    const artifactDir: Record<string, string> = {
      detect: "detections", track: "tracks", ocr: "jerseys",
      "classify-teams": "teams", "court-map": "court", render: "renders",
    };
    const dir = artifactDir[stage];
    if (dir && configKey) {
      await api.deleteArtifact(dir, selectedVideo.id, configKey);
    }
    await handleRunStage(stage);
  };

  return (
    <SidebarProvider>
      <AppSidebar
        videos={videos}
        selectedVideoId={selectedVideoId}
        onVideoSelect={handleVideoSelect}
        onAnalyze={handleAnalyze}
        isRunning={state.status === "running"}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <SidebarInset>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <PipelineStatusBar state={state} />

          {/* Video player area */}
          <div className="aspect-video w-full rounded-lg border bg-card overflow-hidden">
            <VideoCanvas videoId={selectedVideoId} />
          </div>

          {/* Tabbed panels */}
          <AnalysisTabs
            selectedTab={selectedTab}
            onTabChange={setSelectedTab}
            pipelineContent={
              <>
                <PipelineCards state={state} />
                <PipelineTable
                  state={state}
                  staleness={staleness}
                  onRunStage={handleRunStage}
                  onRerunStage={handleRerunStage}
                />
              </>
            }
            playersContent={
              <PlayersTable
                videoId={selectedVideoId}
                pipelineState={state}
                roster={settings.game_context.roster}
              />
            }
            courtContent={<CourtView videoId={selectedVideoId} />}
            chatContent={<ChatPanel />}
          />
        </div>
      </SidebarInset>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </SidebarProvider>
  );
}
