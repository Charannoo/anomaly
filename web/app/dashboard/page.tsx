"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { PlusCircle, ShieldCheck, ArrowRight, RefreshCw } from "lucide-react";
import { MetricValue } from "@/components/MetricValue";
import { InspectionTable } from "@/components/InspectionTable";
import { fetchAnalytics, fetchInspections, AnalyticsData, InspectionListItem } from "@/lib/api";

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [recentInspections, setRecentInspections] = useState<InspectionListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [aData, iData] = await Promise.all([
        fetchAnalytics(),
        fetchInspections({ limit: 10 }),
      ]);
      setAnalytics(aData);
      setRecentInspections(iData.items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-border-default">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-text-primary">
            PNTC Inspect
          </h1>
          <p className="text-xs text-text-muted mt-0.5">
            Multimodal RGB–3D Industrial Inspection & Metrology Core
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={loadData}
            className="p-1.5 rounded-DEFAULT text-text-muted hover:text-text-primary border border-border-default bg-bg-panel hover:bg-bg-surface transition-colors"
            title="Refresh dashboard data"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
          <Link
            href="/inspect"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-DEFAULT bg-accent-primary text-bg-app text-xs font-semibold hover:bg-accent-hover transition-colors shadow-subtle"
          >
            <PlusCircle size={14} />
            <span>New Inspection</span>
          </Link>
        </div>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 border border-border-default bg-bg-panel rounded-md p-4">
        <MetricValue
          label="Total Inspections"
          value={analytics?.total_inspections ?? "—"}
          subtext="Processed samples"
        />
        <MetricValue
          label="Defects Detected"
          value={analytics?.anomalies ?? "—"}
          subtext="Anomalous components"
        />
        <MetricValue
          label="Nominal Parts"
          value={analytics?.normal ?? "—"}
          subtext="Toleranced nominal"
        />
        <MetricValue
          label="Manual Review"
          value={analytics?.manual_review ?? "—"}
          subtext={`${analytics?.manual_review_rate ?? 0}% review guard rate`}
        />
      </div>

      {/* Technical Model Status Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 border border-border-default rounded-md bg-bg-panel p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-border-subtle pb-2.5 mb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-text-secondary">
                  PNTC Architecture Pipeline
                </span>
                <span className="px-1.5 py-0.5 rounded bg-status-normal-bg text-status-normal text-[10px] font-mono font-medium">
                  READY
                </span>
              </div>
              <span className="text-[11px] font-mono text-text-muted">RGB + XYZ Inputs</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
              <div className="bg-bg-subtle p-2.5 rounded border border-border-subtle">
                <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">RGB Backbone</div>
                <div className="font-mono text-text-primary text-[12px] font-medium">DINOv2 ViT-B/14</div>
                <div className="text-[10px] text-text-muted mt-0.5">768-dim tokens</div>
              </div>
              <div className="bg-bg-subtle p-2.5 rounded border border-border-subtle">
                <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Geometry Backbone</div>
                <div className="font-mono text-text-primary text-[12px] font-medium">Point-MAE</div>
                <div className="text-[10px] text-text-muted mt-0.5">1152-dim point patches</div>
              </div>
              <div className="bg-bg-subtle p-2.5 rounded border border-border-subtle col-span-2 sm:col-span-1">
                <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Topology Divergence</div>
                <div className="font-mono text-text-primary text-[12px] font-medium">Jensen-Shannon</div>
                <div className="text-[10px] text-text-muted mt-0.5">k=5, lambda=0.35</div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-2.5 border-t border-border-subtle flex items-center justify-between text-xs">
            <span className="text-text-muted text-[11px]">
              Prototype Coreset Memory: <span className="font-mono text-text-secondary">15,000 paired normals</span>
            </span>
            <Link href="/model" className="inline-flex items-center gap-1 text-accent-primary hover:underline text-[11px]">
              <span>Model Explorer</span>
              <ArrowRight size={12} />
            </Link>
          </div>
        </div>

        {/* Frozen Canonical Benchmark */}
        <div className="border border-border-default rounded-md bg-bg-panel p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-border-subtle pb-2.5 mb-3">
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-text-secondary">
                Frozen Benchmark
              </span>
              <span className="text-[10px] font-mono text-text-muted flex items-center gap-1">
                <ShieldCheck size={11} className="text-status-normal" />
                h5d-pntc-verified
              </span>
            </div>

            <div className="space-y-2.5 font-mono">
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-secondary">Image AUROC</span>
                <span className="text-sm font-semibold tabular-nums text-status-normal">96.541%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-secondary">Pixel AUROC</span>
                <span className="text-sm font-semibold tabular-nums text-status-normal">99.416%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-secondary">AUPRO@0.3</span>
                <span className="text-sm font-semibold tabular-nums text-status-normal">96.939%</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-2.5 border-t border-border-subtle text-[10px] text-text-muted leading-relaxed">
            Preserved verification baseline on MVTec-3D. SOTA multimodal anomaly localization.
          </div>
        </div>
      </div>

      {/* Recent Inspections Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-tight text-text-primary uppercase tracking-wide font-mono">
            Recent Inspections
          </h2>
          <Link
            href="/inspections"
            className="text-xs text-text-muted hover:text-accent-primary transition-colors flex items-center gap-1"
          >
            <span>View All History</span>
            <ArrowRight size={12} />
          </Link>
        </div>
        <InspectionTable inspections={recentInspections} />
      </div>
    </div>
  );
}
