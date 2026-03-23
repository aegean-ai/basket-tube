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
import { downloadVideo, transcribeVideo } from "@/lib/api";
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
  const { state, runPipeline, rerunStage, cancelPipeline, reset, connected } = usePipeline();
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

  const [directRunning, setDirectRunning] = useState<string | null>(null);

  const handleRunStage = async (stage: VisionStage) => {
    if (!selectedVideo) return;
    // Download and transcribe are independent of the vision DAG — call directly
    if (stage === "download") {
      setDirectRunning("download");
      try {
        await downloadVideo(selectedVideo.url === "local" ? selectedVideo.id : selectedVideo.url);
      } finally {
        setDirectRunning(null);
      }
      return;
    }
    if (stage === "transcribe") {
      setDirectRunning("transcribe");
      try {
        await transcribeVideo(selectedVideo.id, settings.stages.transcribe.use_youtube_captions);
      } finally {
        setDirectRunning(null);
      }
      return;
    }
    // Vision stages go through the pipeline orchestrator
    runPipeline(selectedVideo.id, selectedVideo.url, settings, stage);
  };

  const handleRerunStage = (stage: VisionStage) => {
    if (!selectedVideo) return;
    const configKey = state.stages[stage].config_key ?? "";
    rerunStage(stage, selectedVideo.id, configKey, settings);
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
