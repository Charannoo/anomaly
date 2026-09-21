"use client";

import React, { useState, useEffect, useRef } from "react";
import { Send, Sparkles, AlertCircle, ShieldCheck } from "lucide-react";
import { fetchAssistantExplanation, sendAssistantChat } from "@/lib/api";
import { cn } from "@/lib/utils";

interface AssistantPanelProps {
  sampleId: string;
  defectId?: number;
  onActionTriggered?: (actionType: string, meta?: any) => void;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string;
  uiAction?: any;
  sources?: string[];
  latencyMs?: number;
}

export const AssistantPanel: React.FC<AssistantPanelProps> = ({
  sampleId,
  defectId = 1,
  onActionTriggered,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<string>("TECHNICAL");
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load default grounded summary when sampleId changes
  useEffect(() => {
    setLoading(true);
    fetchAssistantExplanation(sampleId, mode)
      .then((res) => {
        setMessages([
          {
            role: "assistant",
            content: res.message,
            model: res.model,
            sources: res.grounding?.fields_used?.slice(0, 4),
            latencyMs: res.latency_ms,
          },
        ]);
      })
      .catch(() => {
        setMessages([
          {
            role: "assistant",
            content: "PNTC AI Assistant ready. Ask any question regarding verified measurements, topology divergence, or geometry diagnostics.",
          },
        ]);
      })
      .finally(() => setLoading(false));
  }, [sampleId, mode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputMessage).trim();
    if (!text || loading) return;

    const userTurn: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userTurn]);
    setInputMessage("");
    setLoading(true);

    try {
      const res = await sendAssistantChat(sampleId, text, undefined, mode, defectId);
      const assistantTurn: Message = {
        role: "assistant",
        content: res.message,
        model: res.model,
        uiAction: res.ui_action,
        sources: res.grounding?.fields_used?.slice(0, 3),
        latencyMs: res.latency_ms,
      };
      setMessages((prev) => [...prev, assistantTurn]);

      if (res.ui_action && onActionTriggered) {
        onActionTriggered(res.ui_action.type, res.ui_action);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Assistant backend communication error. Verified inspection measurements remain fully accessible.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const QUICK_QUESTIONS = [
    "Why was this flagged?",
    "How large is the defect?",
    "What does the 3D evidence show?",
    "Why do RGB and XYZ disagree?",
    "How much material is missing?",
    "Show me the nearest normal example.",
  ];

  return (
    <div className="flex flex-col h-full space-y-3">
      {/* Assistant Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-2">
        <div className="flex items-center gap-1.5 font-mono text-xs text-text-primary">
          <span className="w-2 h-2 rounded-full bg-accent-primary animate-pulse" />
          <span className="font-semibold">PNTC Assistant</span>
        </div>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="bg-bg-app border border-border-subtle rounded text-[11px] font-mono px-2 py-0.5 text-text-secondary outline-none"
        >
          <option value="TECHNICAL">Technical Mode</option>
          <option value="SIMPLE">Operator Mode</option>
          <option value="VIVA">Viva Defense Mode</option>
        </select>
      </div>

      {/* Messages List (Flat industrial text blocks) */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 text-xs font-sans">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "p-2.5 rounded text-xs leading-relaxed",
              msg.role === "user"
                ? "bg-bg-surface border border-border-default ml-4 text-text-primary"
                : "bg-bg-app border-l-2 border-accent-primary pl-3 text-text-secondary"
            )}
          >
            <div className="text-[10px] font-mono text-text-muted mb-1 uppercase font-semibold">
              {msg.role === "user" ? "Operator Query" : `PNTC Assistant ${msg.model ? `(${msg.model})` : ""}`}
            </div>
            <div className="text-text-primary whitespace-pre-wrap">{msg.content}</div>

            {/* UI Action Badge */}
            {msg.uiAction && (
              <div className="mt-2 inline-flex items-center gap-1 px-2 py-0.5 rounded bg-status-normal-bg border border-status-normal/40 text-status-normal font-mono text-[10px]">
                <span>⚡ Action Triggered:</span>
                <span className="font-bold">{msg.uiAction.type}</span>
              </div>
            )}

            {/* Sources Used Trace */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-2 text-[10px] font-mono text-text-muted border-t border-border-subtle/40 pt-1 flex items-center gap-1.5">
                <span>Based on:</span>
                <span className="text-text-secondary">{msg.sources.join(", ")}</span>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="p-2.5 rounded bg-bg-app border-l-2 border-accent-primary pl-3 text-xs text-text-muted font-mono animate-pulse">
            Verifying structured inspection facts...
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Quick Prompts Strip */}
      <div className="flex flex-wrap gap-1.5 pt-1">
        {QUICK_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => handleSend(q)}
            disabled={loading}
            className="px-2 py-1 rounded bg-bg-subtle hover:bg-bg-surface border border-border-subtle hover:border-accent-primary text-[11px] text-text-muted hover:text-text-primary transition-colors text-left"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <div className="flex items-center gap-2 border-t border-border-default pt-2.5">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask about measurements or topology..."
          className="flex-1 bg-bg-app border border-border-default rounded px-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted outline-none focus:border-accent-primary transition-colors font-sans"
        />
        <button
          onClick={() => handleSend()}
          disabled={!inputMessage.trim() || loading}
          className="p-1.5 rounded bg-accent-primary text-bg-app hover:bg-accent-hover transition-colors disabled:opacity-30"
          title="Send message"
        >
          <Send size={13} />
        </button>
      </div>
    </div>
  );
};

export default AssistantPanel;
