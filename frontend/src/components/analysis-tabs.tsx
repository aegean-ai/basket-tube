"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { TabId } from "@/lib/types";

interface AnalysisTabsProps {
  selectedTab: TabId;
  onTabChange: (tab: TabId) => void;
  pipelineContent: React.ReactNode;
  playersContent: React.ReactNode;
  courtContent: React.ReactNode;
  chatContent: React.ReactNode;
}

export function AnalysisTabs({
  selectedTab,
  onTabChange,
  pipelineContent,
  playersContent,
  courtContent,
  chatContent,
}: AnalysisTabsProps) {
  return (
    <Tabs value={selectedTab} onValueChange={(v) => onTabChange(v as TabId)} className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
        <TabsTrigger value="players">Players</TabsTrigger>
        <TabsTrigger value="court">Court</TabsTrigger>
        <TabsTrigger value="chat">Chat</TabsTrigger>
      </TabsList>
      <TabsContent value="pipeline">{pipelineContent}</TabsContent>
      <TabsContent value="players">{playersContent}</TabsContent>
      <TabsContent value="court">{courtContent}</TabsContent>
      <TabsContent value="chat">{chatContent}</TabsContent>
    </Tabs>
  );
}
