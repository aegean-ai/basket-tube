"use client";

import { useState, useEffect } from "react";
import { getArtifact } from "@/lib/api";
import type { PipelineState } from "@/lib/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

interface PlayersTableProps {
  videoId?: string;
  pipelineState: PipelineState;
  roster: Record<string, string>;
  onSeekToPlayer?: (trackerIdOrTime: number) => void;
}

interface JerseysArtifact {
  players: Record<string, string>; // tracker_id -> jersey number
}

interface TeamsArtifact {
  palette: Record<string, { name: string; color: string }>; // team_id -> { name, color }
  assignments?: Record<string, string>; // tracker_id -> team_id
}

interface PlayerRow {
  trackerId: string;
  jerseyNumber: string;
  playerName: string;
  teamId: string | null;
  teamName: string;
  teamColor: string;
}

export function PlayersTable({
  videoId,
  pipelineState,
  roster,
  onSeekToPlayer,
}: PlayersTableProps) {
  const [jerseys, setJerseys] = useState<JerseysArtifact | null>(null);
  const [teams, setTeams] = useState<TeamsArtifact | null>(null);

  const ocrStage = pipelineState.stages["ocr"];
  const classifyStage = pipelineState.stages["classify-teams"];
  const ocrComplete = ocrStage?.status === "complete";
  const ocrConfigKey = ocrStage?.config_key;
  const classifyConfigKey = classifyStage?.config_key;

  useEffect(() => {
    if (!ocrComplete || !videoId || !ocrConfigKey) return;

    getArtifact("jerseys", videoId, ocrConfigKey)
      .then((data) => setJerseys(data as JerseysArtifact))
      .catch(console.error);
  }, [ocrComplete, videoId, ocrConfigKey]);

  useEffect(() => {
    if (!ocrComplete || !videoId || !classifyConfigKey) return;

    getArtifact("teams", videoId, classifyConfigKey)
      .then((data) => setTeams(data as TeamsArtifact))
      .catch(console.error);
  }, [ocrComplete, videoId, classifyConfigKey]);

  if (!ocrComplete) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground text-sm">
        Run pipeline to detect players
      </div>
    );
  }

  const players: PlayerRow[] = jerseys
    ? Object.entries(jerseys.players).map(([trackerId, jerseyNumber]) => {
        const playerName = roster[jerseyNumber] ?? "-";
        const teamId = teams?.assignments?.[trackerId] ?? null;
        const teamInfo = teamId && teams?.palette?.[teamId];
        return {
          trackerId,
          jerseyNumber,
          playerName,
          teamId,
          teamName: teamInfo ? teamInfo.name : "-",
          teamColor: teamInfo ? teamInfo.color : "#888888",
        };
      })
    : [];

  if (players.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground text-sm">
        No players detected
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Jersey #</TableHead>
          <TableHead>Player Name</TableHead>
          <TableHead>Team</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {players.map((player) => (
          <TableRow
            key={player.trackerId}
            className={onSeekToPlayer ? "cursor-pointer" : undefined}
            onClick={() => onSeekToPlayer?.(Number(player.trackerId))}
          >
            <TableCell className="font-mono font-medium">
              {player.jerseyNumber}
            </TableCell>
            <TableCell>{player.playerName}</TableCell>
            <TableCell>
              <Badge
                style={{
                  backgroundColor: player.teamColor,
                  color: "#ffffff",
                  borderColor: "transparent",
                }}
              >
                {player.teamName}
              </Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
