"use client";

import { Dialog } from "@base-ui/react/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { XIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { useAnalysisSettings } from "@/contexts/analysis-settings-context";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { settings, updateGameContext, updateAdvanced } = useAnalysisSettings();
  const { game_context, advanced } = settings;

  // Team helpers
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

  // Roster helpers
  const rosterEntries = Object.entries(game_context.roster);

  const addPlayer = () => {
    // Find an unused jersey number placeholder
    let jersey = "";
    let n = rosterEntries.length + 1;
    while (game_context.roster[String(n)] !== undefined) n++;
    jersey = String(n);
    updateGameContext({
      roster: { ...game_context.roster, [jersey]: "" },
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
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/50 transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0 supports-backdrop-filter:backdrop-blur-sm" />
        <Dialog.Popup className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-background shadow-2xl transition duration-200 data-ending-style:opacity-0 data-ending-style:scale-95 data-starting-style:opacity-0 data-starting-style:scale-95 w-[560px] max-h-[85vh] overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div>
              <Dialog.Title className="text-lg font-semibold">
                Analysis Settings
              </Dialog.Title>
              <Dialog.Description className="text-sm text-muted-foreground mt-1">
                Configure game context and pipeline parameters.
              </Dialog.Description>
            </div>

            <Separator />

            {/* Game Context */}
            <section className="space-y-4">
              <h3 className="text-sm font-semibold">Game Context</h3>

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
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={addPlayer}
                    className="gap-1.5"
                  >
                    <PlusIcon className="size-3.5" />
                    Add Player
                  </Button>
                </div>

                {rosterEntries.length === 0 ? (
                  <p className="text-xs text-muted-foreground py-2">
                    No players added yet. Click "Add Player" to start.
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
                              onChange={(e) =>
                                updatePlayerJersey(jersey, e.target.value)
                              }
                              className="w-20 font-mono"
                            />
                          </TableCell>
                          <TableCell>
                            <Input
                              value={name}
                              onChange={(e) =>
                                updatePlayerName(jersey, e.target.value)
                              }
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
            </section>

            {/* Advanced (collapsible) */}
            <Accordion>
              <AccordionItem value="advanced">
                <AccordionTrigger className="text-sm font-semibold">
                  Advanced
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4 pt-2">
                    {/* Detection Confidence */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <label className="text-sm">Detection Confidence</label>
                        <span className="text-xs text-muted-foreground font-mono">
                          {advanced.confidence.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={advanced.confidence}
                        onChange={(e) =>
                          updateAdvanced({ confidence: Number(e.target.value) })
                        }
                        className="w-full accent-primary"
                      />
                    </div>

                    {/* IOU Threshold */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <label className="text-sm">IOU Threshold</label>
                        <span className="text-xs text-muted-foreground font-mono">
                          {advanced.iou_threshold.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={advanced.iou_threshold}
                        onChange={(e) =>
                          updateAdvanced({
                            iou_threshold: Number(e.target.value),
                          })
                        }
                        className="w-full accent-primary"
                      />
                    </div>

                    {/* OCR Interval */}
                    <div className="flex items-center justify-between gap-4">
                      <label className="text-sm">OCR Interval (frames)</label>
                      <Input
                        type="number"
                        min={1}
                        value={advanced.ocr_interval}
                        onChange={(e) =>
                          updateAdvanced({
                            ocr_interval: Number(e.target.value),
                          })
                        }
                        className="w-24 text-right font-mono"
                      />
                    </div>

                    {/* Team Crop Scale */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <label className="text-sm">Team Crop Scale</label>
                        <span className="text-xs text-muted-foreground font-mono">
                          {advanced.crop_scale.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={advanced.crop_scale}
                        onChange={(e) =>
                          updateAdvanced({ crop_scale: Number(e.target.value) })
                        }
                        className="w-full accent-primary"
                      />
                    </div>

                    {/* Sampling Stride */}
                    <div className="flex items-center justify-between gap-4">
                      <label className="text-sm">Sampling Stride (frames)</label>
                      <Input
                        type="number"
                        min={1}
                        value={advanced.stride}
                        onChange={(e) =>
                          updateAdvanced({ stride: Number(e.target.value) })
                        }
                        className="w-24 text-right font-mono"
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
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
