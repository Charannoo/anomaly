"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { GitCompare, ArrowRight } from "lucide-react";
import { fetchInspections, fetchInspection, InspectionListItem, InspectionDetail } from "@/lib/api";
import { StatusIndicator } from "@/components/StatusIndicator";
import { formatMetric } from "@/lib/utils";

export default function ComparePage() {
  const [inspectionsList, setInspectionsList] = useState<InspectionListItem[]>([]);
  const [leftId, setLeftId] = useState<string>("INSP-01_strong_defect");
  const [rightId, setRightId] = useState<string>("INSP-07_nominal_sample");
  const [leftDetail, setLeftDetail] = useState<InspectionDetail | null>(null);
  const [rightDetail, setRightDetail] = useState<InspectionDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchInspections({ limit: 50 }).then((res) => {
      setInspectionsList(res.items);
      if (res.items.length >= 2) {
        setLeftId(res.items[0].id);
        setRightId(res.items[1].id);
      }
    });
  }, []);

  useEffect(() => {
    if (leftId && rightId) {
      setLoading(true);
      Promise.all([fetchInspection(leftId), fetchInspection(rightId)])
        .then(([l, r]) => {
          setLeftDetail(l);
          setRightDetail(r);
        })
        .finally(() => setLoading(false));
    }
  }, [leftId, rightId]);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-border-default flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-text-primary">
            Compare Inspections Side-by-Side
          </h1>
          <p className="text-xs text-text-muted mt-0.5">
            Bilateral cross-examination of anomalous vs nominal samples or evolving defect geometries.
          </p>
        </div>
      </div>

      {/* Selectors Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border border-border-default rounded-md bg-bg-panel p-3 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="text-text-muted text-[11px] whitespace-nowrap">Sample A:</span>
          <select
            value={leftId}
            onChange={(e) => setLeftId(e.target.value)}
            className="w-full bg-bg-app border border-border-default rounded px-2.5 py-1 text-xs text-text-primary outline-none"
          >
            {inspectionsList.map((i) => (
              <option key={`left-${i.id}`} value={i.id}>
                {i.id} — {i.category} ({i.status})
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-text-muted text-[11px] whitespace-nowrap">Sample B:</span>
          <select
            value={rightId}
            onChange={(e) => setRightId(e.target.value)}
            className="w-full bg-bg-app border border-border-default rounded px-2.5 py-1 text-xs text-text-primary outline-none"
          >
            {inspectionsList.map((i) => (
              <option key={`right-${i.id}`} value={i.id}>
                {i.id} — {i.category} ({i.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Comparison Workspace */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[leftDetail, rightDetail].map((detail, idx) => {
          if (!detail) {
            return (
              <div key={idx} className="p-12 border border-border-default rounded bg-bg-panel text-center text-xs text-text-muted font-mono">
                Loading sample...
              </div>
            );
          }

          const rep = detail.report || {};
          const pntc = rep.pntc || {};
          const d0 = rep.defects?.[0] || {};
          const morph = d0.morphology || {};
          const geom = d0.geometry || {};
          const vol = d0.volume || {};

          return (
            <div key={detail.id} className="border border-border-default rounded-md bg-bg-panel p-4 space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between border-b border-border-subtle pb-2.5">
                <div>
                  <span className="text-[10px] font-mono text-text-muted uppercase block">
                    Sample {idx === 0 ? "A" : "B"}
                  </span>
                  <Link href={`/inspect/${detail.id}`} className="text-sm font-semibold text-accent-primary hover:underline">
                    {detail.id}
                  </Link>
                  <span className="text-xs text-text-muted ml-2">({detail.category})</span>
                </div>
                <StatusIndicator status={detail.status} size="sm" />
              </div>

              {/* Visual Overlay Image */}
              <div className="h-60 rounded bg-bg-app border border-border-subtle overflow-hidden flex items-center justify-center">
                <img
                  src={`/api/inspections/${detail.id}/artifacts/overlay`}
                  alt="Inspection Overlay"
                  className="max-h-full max-w-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = `/api/inspections/${detail.id}/artifacts/rgb`;
                  }}
                />
              </div>

              {/* Core Quantitative Metrics */}
              <div className="border border-border-subtle rounded bg-bg-subtle p-3 space-y-2 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">PNTC Anomaly Score</span>
                  <span className="font-bold text-text-primary tabular-nums text-sm">
                    {formatMetric(pntc.score, 4)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Defects Detected</span>
                  <span className="font-semibold text-text-primary tabular-nums">
                    {rep.defects?.length ?? 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Max Depression Depth</span>
                  <span className="text-status-anomaly font-semibold">
                    {geom.max_depression_mm !== undefined && geom.max_depression_mm !== null
                      ? `${formatMetric(geom.max_depression_mm, 2)} mm`
                      : "0.00 mm"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Missing Material Volume</span>
                  <span className="text-status-anomaly font-semibold">
                    {vol.missing_material !== undefined
                      ? `${formatMetric(vol.missing_material, 1)} mm³`
                      : "0.0 mm³"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Surface Area</span>
                  <span className="text-text-primary">
                    {morph.surface_area_3d_mm2 !== undefined
                      ? `${formatMetric(morph.surface_area_3d_mm2, 1)} mm²`
                      : "—"}
                  </span>
                </div>
              </div>

              <div className="pt-2 text-right">
                <Link
                  href={`/inspect/${detail.id}`}
                  className="inline-flex items-center gap-1 text-xs text-accent-primary hover:underline font-mono"
                >
                  <span>Open Full Metrology Report</span>
                  <ArrowRight size={12} />
                </Link>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
