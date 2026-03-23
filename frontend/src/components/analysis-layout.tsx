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
  const { state, runPipeline, runStage, rerunStage, cancelPipeline, reset, connected } = usePipeline();
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

  // Build the params each stage endpoint expects from current settings
  const stageParams = (stage: VisionStage): object => {
    const s = settings.stages;
    const detKey = state.stages.detect.config_key ?? "";
    const trackKey = state.stages.track.config_key ?? "";
    switch (stage) {
      case "detect":
        return { model_id: s.detect.model_id, confidence: s.detect.confidence, iou_threshold: s.detect.iou_threshold };
      case "track":
        return { det_config_key: detKey };
      case "ocr":
        return { track_config_key: trackKey, ocr_interval: s.ocr.ocr_interval };
      case "classify-teams":
        return { det_config_key: detKey, stride: s.teams.stride, crop_scale: s.teams.crop_scale };
      case "court-map":
        return { det_config_key: detKey };
      default:
        return {};
    }
  };

  const handleRunStage = (stage: VisionStage) => {
    if (!selectedVideoId) return;
    runStage(stage, selectedVideoId, stageParams(stage));
  };

  const handleRerunStage = (stage: VisionStage) => {
    if (!selectedVideoId) return;
    const configKey = state.stages[stage].config_key ?? "";
    rerunStage(stage, selectedVideoId, configKey, stageParams(stage));
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
