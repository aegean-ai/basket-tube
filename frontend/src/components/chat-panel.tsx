"use client";

import { useState, useRef, useEffect } from "react";
import type { ChatMessage } from "@/lib/types";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { SendIcon } from "lucide-react";

const ASSISTANT_PLACEHOLDER =
  "Chat backend coming soon — this will use AI to answer questions about the game.";

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = draft.trim();
    if (!text) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    const assistantMsg: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: ASSISTANT_PLACEHOLDER,
      timestamp: Date.now() + 1,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setDraft("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className="flex flex-col min-h-[400px]">
      <CardContent className="flex-1 p-0">
        <ScrollArea className="h-[340px] px-4 py-3">
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-8">
              Ask a question about the game
            </p>
          )}
          <div className="flex flex-col gap-3">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[75%] rounded-xl px-3 py-2 text-sm ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
      </CardContent>
      <CardFooter className="gap-2">
        <Input
          placeholder="Ask about the game..."
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1"
        />
        <Button
          size="icon"
          onClick={handleSend}
          disabled={!draft.trim()}
          aria-label="Send message"
        >
          <SendIcon className="size-4" />
        </Button>
      </CardFooter>
    </Card>
  );
}
