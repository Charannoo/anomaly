"use client";

import React, { useEffect, useState } from "react";
import { fetchProviderStatus, fetchSystemStatus } from "@/lib/api";
import { Shield, Database, FileText, CheckCircle2, RefreshCw } from "lucide-react";

export default function SettingsPage() {
  const [providerStatus, setProviderStatus] = useState<any>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [reportFormat, setReportFormat] = useState("html");
  const [defaultMode, setDefaultMode] = useState("TECHNICAL");
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const loadStatus = async () => {
    setLoading(true);
    try {
      const [prov, sys] = await Promise.all([
        fetchProviderStatus().catch(() => null),
        fetchSystemStatus().catch(() => null),
      ]);
      setProviderStatus(prov);
      setSystemStatus(sys);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const handleSwitchProvider = async (provider: string) => {
    setSwitching(true);
    try {
      const res = await fetch("/api/assistant/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });
      if (res.ok) {
        const updated = await res.json();
        setProviderStatus(updated);
        setSaveMessage(`Active provider switched to ${provider.toUpperCase()}`);
        setTimeout(() => setSaveMessage(null), 3000);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-6 border-b border-white/[0.05]">
        <div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-text-primary">
            Settings
          </h1>
          <p className="text-sm text-text-muted mt-1">
            Inspection runtime environments, assistant providers, and reporting configurations.
          </p>
        </div>
        <button
          onClick={loadStatus}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-xs font-medium text-text-primary transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      </div>

      {saveMessage && (
        <div className="p-3.5 rounded bg-status-normal/10 border border-status-normal/30 text-xs text-status-normal flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" />
          <span>{saveMessage}</span>
        </div>
      )}

      {/* Section 1: AI Assistant Provider */}
      <div className="p-6 rounded-md border border-white/[0.06] bg-bg-surface space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Shield className="w-4 h-4 text-accent-primary" />
            <h2 className="text-base font-semibold text-text-primary">Assistant provider</h2>
          </div>
          <span className="text-xs text-text-muted">Zero secret exposure</span>
        </div>

        <p className="text-xs text-text-muted leading-relaxed">
          The assistant synthesizes verified geometric and topological evidence into natural language. API keys are
          managed securely via server environment variables and are never transmitted to the browser client.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
          {/* Gemini Option */}
          <div
            className={`p-4 rounded border transition-all ${
              providerStatus?.provider_name === "gemini"
                ? "border-accent-primary/60 bg-accent-primary/5"
                : "border-white/[0.06] bg-bg-app"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-text-primary">Google Gemini (Primary)</span>
              {providerStatus?.gemini_configured ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-status-normal/10 text-status-normal font-medium">
                  <CheckCircle2 className="w-3 h-3" /> Configured
                </span>
              ) : (
                <span className="text-[11px] text-text-muted bg-white/[0.04] px-2 py-0.5 rounded">
                  Not configured
                </span>
              )}
            </div>
            <p className="text-xs text-text-muted mb-3">
              Primary provider for structured metrology grounding. Model: gemini-1.5-flash.
            </p>
            <button
              onClick={() => handleSwitchProvider("gemini")}
              disabled={switching || providerStatus?.provider_name === "gemini"}
              className="w-full py-1.5 px-3 rounded text-xs font-medium border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-text-primary disabled:opacity-40 transition-colors"
            >
              {providerStatus?.provider_name === "gemini" ? "Active provider" : "Switch to Gemini"}
            </button>
          </div>

          {/* Grok Option */}
          <div
            className={`p-4 rounded border transition-all ${
              providerStatus?.provider_name === "grok"
                ? "border-accent-primary/60 bg-accent-primary/5"
                : "border-white/[0.06] bg-bg-app"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-text-primary">xAI Grok (Alternative)</span>
              {providerStatus?.grok_configured ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-status-normal/10 text-status-normal font-medium">
                  <CheckCircle2 className="w-3 h-3" /> Configured
                </span>
              ) : (
                <span className="text-[11px] text-text-muted bg-white/[0.04] px-2 py-0.5 rounded">
                  Not configured
                </span>
              )}
            </div>
            <p className="text-xs text-text-muted mb-3">
              xAI reasoning fallback provider. Model: grok-beta.
            </p>
            <button
              onClick={() => handleSwitchProvider("grok")}
              disabled={switching || providerStatus?.provider_name === "grok"}
              className="w-full py-1.5 px-3 rounded text-xs font-medium border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-text-primary disabled:opacity-40 transition-colors"
            >
              {providerStatus?.provider_name === "grok" ? "Active provider" : "Switch to Grok"}
            </button>
          </div>
        </div>
      </div>

      {/* Section 2: Data Persistence & Storage */}
      <div className="p-6 rounded-md border border-white/[0.06] bg-bg-surface space-y-4">
        <div className="flex items-center gap-2.5">
          <Database className="w-4 h-4 text-accent-primary" />
          <h2 className="text-base font-semibold text-text-primary">Data persistence & storage engine</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded bg-bg-app border border-white/[0.05]">
            <div className="text-xs text-text-muted">Active database</div>
            <div className="text-text-primary font-semibold mt-1">SQLite 3 (WAL Mode)</div>
            <div className="text-xs text-text-muted font-mono mt-1">data/inspections.db</div>
          </div>

          <div className="p-4 rounded bg-bg-app border border-white/[0.05]">
            <div className="text-xs text-text-muted">Artifacts & metrology storage</div>
            <div className="text-text-primary font-semibold mt-1">Local filesystem</div>
            <div className="text-xs text-text-muted font-mono mt-1">results/demo_cases/</div>
          </div>
        </div>
      </div>

      {/* Section 3: Report & Metrology Preferences */}
      <div className="p-6 rounded-md border border-white/[0.06] bg-bg-surface space-y-4">
        <div className="flex items-center gap-2.5">
          <FileText className="w-4 h-4 text-accent-primary" />
          <h2 className="text-base font-semibold text-text-primary">Export & documentation preferences</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="block text-text-muted mb-1.5 font-medium">Default certificate format</label>
            <select
              value={reportFormat}
              onChange={(e) => setReportFormat(e.target.value)}
              className="w-full px-3 py-2 rounded bg-bg-app border border-white/[0.08] text-text-primary text-xs focus:outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="html">ISO Metrology Certificate (HTML / PDF Print)</option>
              <option value="json">Raw Scientific JSON Schema</option>
              <option value="txt">ASCII Metrology Summary (TXT)</option>
            </select>
          </div>

          <div>
            <label className="block text-text-muted mb-1.5 font-medium">Default explanation mode</label>
            <select
              value={defaultMode}
              onChange={(e) => setDefaultMode(e.target.value)}
              className="w-full px-3 py-2 rounded bg-bg-app border border-white/[0.08] text-text-primary text-xs focus:outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="TECHNICAL">Technical (Quantitative measurements & divergence)</option>
              <option value="SIMPLE">Simple (Operator-oriented plain language)</option>
              <option value="VIVA">Viva / Defense (Formal academic justification)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section 4: System Freeze Audit */}
      <div className="p-5 rounded-md border border-white/[0.05] bg-bg-surface flex items-center justify-between text-xs">
        <div>
          <div className="font-semibold text-text-primary">PNTC core model status</div>
          <div className="text-xs text-text-muted mt-0.5">
            Detector metrics and backbones are frozen and sealed. No online training or parameter updates permitted.
          </div>
        </div>
        <div className="px-3 py-1.5 rounded bg-status-normal/10 border border-status-normal/30 text-status-normal font-mono text-xs font-semibold shrink-0">
          SEALED (AUROC: 96.541%)
        </div>
      </div>
    </div>
  );
}
