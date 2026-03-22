"use client";

import { useState, useEffect } from "react";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { AnalysisTabs } from "@/components/analysis-tabs";
import { PipelineStatusBar } from "@/components/pipeline-status-bar";
import { usePipeline } from "@/hooks/use-pipeline";
import { useAnalysisSettings } from "@/contexts/analysis-settings-context";
import type { Video, TabId } from "@/lib/types";

interface AnalysisLayoutProps {
  videos: Video[];
}

export function AnalysisLayout({ videos }: AnalysisLayoutProps) {
  const [selectedVideoId, setSelectedVideoId] = useState<string | undefined>(
    videos[0]?.id
  );
  const [selectedTab, setSelectedTab] = useState<TabId>("pipeline");
  const { state, runPipeline, reset } = usePipeline();
  const { settings, loadForVideo } = useAnalysisSettings();

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

  return (
    <SidebarProvider>
      <AppSidebar
        videos={videos}
        selectedVideoId={selectedVideoId}
        onVideoSelect={handleVideoSelect}
        onAnalyze={handleAnalyze}
        isRunning={state.status === "running"}
      />
      <SidebarInset>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <PipelineStatusBar state={state} />

          {/* Video player area - placeholder until FE-8 */}
          <div className="aspect-video w-full rounded-lg border bg-card flex items-center justify-center text-muted-foreground">
            {selectedVideo ? selectedVideo.title : "Select a video"}
          </div>

          {/* Tabbed panels */}
          <AnalysisTabs
            selectedTab={selectedTab}
            onTabChange={setSelectedTab}
            pipelineContent={
              <div className="p-4 text-muted-foreground">Pipeline table — coming in FE-7</div>
            }
            playersContent={
              <div className="p-4 text-muted-foreground">Players table — coming in FE-9</div>
            }
            courtContent={
              <div className="p-4 text-muted-foreground">Court view — coming in FE-10</div>
            }
            chatContent={
              <div className="p-4 text-muted-foreground">Chat — coming in FE-10</div>
            }
          />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
