"use client";

import * as React from "react";
import { CircleDot, VideoIcon, PlayIcon, SettingsIcon, WorkflowIcon, UsersIcon, MapIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import type { Video } from "@/lib/types";

export type AnalyticsView = "pipeline" | "players" | "court";

const ANALYTICS_VIEWS: { key: AnalyticsView; label: string; icon: React.ElementType }[] = [
  { key: "pipeline", label: "Pipeline", icon: WorkflowIcon },
  { key: "players", label: "Players", icon: UsersIcon },
  { key: "court", label: "Court", icon: MapIcon },
];

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  videos: Video[];
  selectedVideoId?: string;
  onVideoSelect: (id: string) => void;
  onAnalyze: () => void;
  isRunning: boolean;
  onOpenSettings?: () => void;
  analyticsView: AnalyticsView;
  onAnalyticsViewChange: (view: AnalyticsView) => void;
}

export function AppSidebar({
  videos,
  selectedVideoId,
  onVideoSelect,
  onAnalyze,
  isRunning,
  onOpenSettings,
  analyticsView,
  onAnalyticsViewChange,
  ...props
}: AppSidebarProps) {
  return (
    <Sidebar {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" render={<div />}>
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <CircleDot className="size-4" />
              </div>
              <div className="flex flex-col gap-0.5 leading-none">
                <span className="font-semibold">BasketTube</span>
                <span className="text-xs text-muted-foreground">aegean.ai</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Video Library</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {videos.map((video) => {
                const isActive = video.id === selectedVideoId;
                return (
                  <SidebarMenuItem key={video.id}>
                    <SidebarMenuButton
                      isActive={isActive}
                      onClick={() => onVideoSelect(video.id)}
                      tooltip={video.title}
                      className={`h-auto py-1.5 ${isActive ? "border-l-2 border-primary bg-sidebar-accent/80 pl-1.5" : ""}`}
                    >
                      <VideoIcon className="mt-0.5 shrink-0" />
                      <div className="flex flex-col min-w-0">
                        <span className="text-sm leading-snug">{video.title}</span>
                        <span className="text-[10px] text-muted-foreground font-mono">{video.id}</span>
                      </div>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Analytics</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {ANALYTICS_VIEWS.map(({ key, label, icon: Icon }) => (
                <SidebarMenuItem key={key}>
                  <SidebarMenuButton
                    isActive={analyticsView === key}
                    onClick={() => onAnalyticsViewChange(key)}
                    tooltip={label}
                  >
                    <Icon className="shrink-0" />
                    <span>{label}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="flex gap-2">
          <Button
            className="flex-1"
            onClick={onAnalyze}
            disabled={isRunning}
          >
            <PlayIcon className="size-3.5 mr-1.5" />
            {isRunning ? "Analyzing..." : "Analyze"}
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={onOpenSettings}
            aria-label="Open settings"
          >
            <SettingsIcon className="size-4" />
          </Button>
        </div>
        <div className="text-center text-[10px] text-muted-foreground/60 pb-1">
          aegean.ai
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
