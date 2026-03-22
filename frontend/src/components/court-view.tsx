"use client";

import { Card, CardContent } from "@/components/ui/card";

interface CourtViewProps {
  videoId?: string;
}

export function CourtView({ videoId: _videoId }: CourtViewProps) {
  return (
    <Card className="min-h-[320px]">
      <CardContent className="flex flex-col items-center justify-center min-h-[320px] gap-3 text-muted-foreground">
        <span className="text-5xl" aria-hidden="true">🏀</span>
        <p className="text-sm font-medium">Court visualization coming soon</p>
      </CardContent>
    </Card>
  );
}
