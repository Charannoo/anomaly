"use client";

import React, { useEffect, useState } from "react";
import { fetchProviderStatus, fetchSystemStatus } from "@/lib/api";
import { Sliders, Shield, Database, FileText, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

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
    <div className="space-y-6 max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F7]">Platform Settings</h1>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Inspection runtime environments, assistant providers, and reporting configurations.
            </p>
          </div>
          <button
            onClick={loadStatus}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/[0.08] bg-[#15191F] hover:bg-[#191E25] text-xs font-medium text-[#F3F5F7] transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {saveMessage && (
          <div className="p-3 rounded-lg bg-[#55B98A]/10 border border-[#55B98A]/30 text-xs text-[#55B98A] flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            {saveMessage}
          </div>
        )}

        {/* Section 1: AI Assistant Provider */}
        <div className="p-5 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-[#5BB8C4]" />
              <h2 className="text-sm font-semibold text-[#F3F5F7]">AI Assistant Provider Configuration</h2>
            </div>
            <span className="text-[11px] text-[#6F7884] font-mono">ZERO SECRET EXPOSURE</span>
          </div>

          <p className="text-xs text-[#A7AFBA]">
            PNTC Assistant translates verified geometric and topological evidence into natural language. API keys are
            managed securely via environment variables and are never surfaced to the browser client.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Gemini Option */}
            <div
              className={`p-4 rounded-md border transition-all ${
                providerStatus?.provider_name === "gemini"
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/5"
                  : "border-white/[0.08] bg-[#101318]"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-[#F3F5F7]">Google Gemini (Primary)</span>
                {providerStatus?.gemini_configured ? (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-[#55B98A]/10 text-[#55B98A] border border-[#55B98A]/30 font-medium">
                    <CheckCircle2 className="w-3 h-3" /> Configured
                  </span>
                ) : (
                  <span className="text-[10px] text-[#A7AFBA] bg-white/[0.04] px-2 py-0.5 rounded">
                    Not configured
                  </span>
                )}
              </div>
              <p className="text-[11px] text-[#A7AFBA] mb-3">
                Default provider for structured metrology grounding. Model: gemini-1.5-flash / pro.
              </p>
              <button
                onClick={() => handleSwitchProvider("gemini")}
                disabled={switching || providerStatus?.provider_name === "gemini"}
                className="w-full py-1.5 px-3 rounded text-xs font-medium border border-white/[0.08] bg-[#15191F] hover:bg-[#191E25] text-[#F3F5F7] disabled:opacity-40 transition-colors"
              >
                {providerStatus?.provider_name === "gemini" ? "Active Provider" : "Switch to Gemini"}
              </button>
            </div>

            {/* Grok Option */}
            <div
              className={`p-4 rounded-md border transition-all ${
                providerStatus?.provider_name === "grok"
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/5"
                  : "border-white/[0.08] bg-[#101318]"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-[#F3F5F7]">xAI Grok (Alternative)</span>
                {providerStatus?.grok_configured ? (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-[#55B98A]/10 text-[#55B98A] border border-[#55B98A]/30 font-medium">
                    <CheckCircle2 className="w-3 h-3" /> Configured
                  </span>
                ) : (
                  <span className="text-[10px] text-[#A7AFBA] bg-white/[0.04] px-2 py-0.5 rounded">
                    Not configured
                  </span>
                )}
              </div>
              <p className="text-[11px] text-[#A7AFBA] mb-3">
                xAI reasoning backbone. Model: grok-beta. Fallback provider.
              </p>
              <button
                onClick={() => handleSwitchProvider("grok")}
                disabled={switching || providerStatus?.provider_name === "grok"}
                className="w-full py-1.5 px-3 rounded text-xs font-medium border border-white/[0.08] bg-[#15191F] hover:bg-[#191E25] text-[#F3F5F7] disabled:opacity-40 transition-colors"
              >
                {providerStatus?.provider_name === "grok" ? "Active Provider" : "Switch to Grok"}
              </button>
            </div>
          </div>
        </div>

        {/* Section 2: Data Persistence & Storage */}
        <div className="p-5 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-[#5BB8C4]" />
            <h2 className="text-sm font-semibold text-[#F3F5F7]">Data Persistence & Storage Engine</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[10px] text-[#6F7884] font-sans">Active Database</div>
              <div className="text-[#F3F5F7] font-semibold mt-0.5">SQLite 3 (WAL Mode)</div>
              <div className="text-[11px] text-[#A7AFBA] mt-1 break-all">data/inspections.db</div>
            </div>

            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[10px] text-[#6F7884] font-sans">Artifacts & Plotly Storage</div>
              <div className="text-[#F3F5F7] font-semibold mt-0.5">Local Filesystem</div>
              <div className="text-[11px] text-[#A7AFBA] mt-1 break-all">results/demo_cases/</div>
            </div>
          </div>
        </div>

        {/* Section 3: Report & Metrology Preferences */}
        <div className="p-5 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-[#5BB8C4]" />
            <h2 className="text-sm font-semibold text-[#F3F5F7]">Export & Documentation Preferences</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block text-[#A7AFBA] mb-1.5 font-medium">Default Certificate Export Format</label>
              <select
                value={reportFormat}
                onChange={(e) => setReportFormat(e.target.value)}
                className="w-full px-3 py-2 rounded bg-[#101318] border border-white/[0.08] text-[#F3F5F7] text-xs focus:outline-none focus:border-[#5BB8C4]"
              >
                <option value="html">ISO Metrology Certificate (HTML / PDF Print)</option>
                <option value="json">Raw Scientific JSON Schema</option>
                <option value="txt">ASCII Metrology Summary (TXT)</option>
              </select>
            </div>

            <div>
              <label className="block text-[#A7AFBA] mb-1.5 font-medium">Default Explanation Mode</label>
              <select
                value={defaultMode}
                onChange={(e) => setDefaultMode(e.target.value)}
                className="w-full px-3 py-2 rounded bg-[#101318] border border-white/[0.08] text-[#F3F5F7] text-xs focus:outline-none focus:border-[#5BB8C4]"
              >
                <option value="TECHNICAL">Technical (Quantitative measurements & divergence)</option>
                <option value="SIMPLE">Simple (Operator-oriented plain language)</option>
                <option value="VIVA">Viva / Defense (Formal academic justification)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 4: System Freeze Audit */}
        <div className="p-5 rounded-lg border border-white/[0.08] bg-[#101318] flex items-center justify-between text-xs">
          <div>
            <div className="font-semibold text-[#F3F5F7]">PNTC Core Model Status</div>
            <div className="text-[11px] text-[#A7AFBA] mt-0.5">
              Detector metrics and backbones are frozen and sealed. No online training or parameter updates permitted.
            </div>
          </div>
          <div className="px-3 py-1.5 rounded bg-[#55B98A]/10 border border-[#55B98A]/30 text-[#55B98A] font-mono text-xs font-semibold shrink-0">
            SEALED (AUROC: 96.541%)
          </div>
        </div>
      </div>
  );
}
