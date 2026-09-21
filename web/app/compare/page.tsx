"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
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
    <div className="max-w-[1500px] mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-6 border-b border-white/[0.05]">
        <div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-text-primary">
            Compare
          </h1>
          <p className="text-sm text-text-muted mt-1">
            Side-by-side multimodal cross-examination of sample geometries and defect signatures.
          </p>
        </div>
      </div>

      {/* Selectors Bar - Clean strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
        <div className="flex items-center gap-3">
          <span className="text-text-muted text-xs font-medium shrink-0">Sample A</span>
          <select
            value={leftId}
            onChange={(e) => setLeftId(e.target.value)}
            className="w-full bg-bg-surface border border-white/[0.08] rounded px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-primary cursor-pointer"
          >
            {inspectionsList.map((i) => (
              <option key={`left-${i.id}`} value={i.id}>
                {i.id} — {i.category} ({i.status})
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-text-muted text-xs font-medium shrink-0">Sample B</span>
          <select
            value={rightId}
            onChange={(e) => setRightId(e.target.value)}
            className="w-full bg-bg-surface border border-white/[0.08] rounded px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-primary cursor-pointer"
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {[leftDetail, rightDetail].map((detail, idx) => {
          if (!detail) {
            return (
              <div
                key={idx}
                className="p-16 border border-white/[0.05] rounded-md bg-bg-surface text-center text-xs text-text-muted"
              >
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
            <div
              key={detail.id}
              className="border border-white/[0.06] rounded-md bg-bg-surface p-5 space-y-5"
            >
              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-white/[0.05] pb-3">
                <div>
                  <span className="text-[11px] text-text-muted block">
                    Sample {idx === 0 ? "A" : "B"}
                  </span>
                  <Link
                    href={`/inspect/${detail.id}`}
                    className="text-base font-semibold text-text-primary hover:text-accent-primary transition-colors"
                  >
                    {detail.id}
                  </Link>
                  <span className="text-xs text-text-muted ml-2">({detail.category})</span>
                </div>
                <StatusIndicator status={detail.status} size="sm" />
              </div>

              {/* Visual Overlay Image */}
              <div className="h-64 rounded bg-bg-app border border-white/[0.04] overflow-hidden flex items-center justify-center">
                <img
                  src={`/api/inspections/${detail.id}/artifacts/overlay`}
                  alt="Inspection Overlay"
                  className="max-h-full max-w-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = `/api/inspections/${detail.id}/artifacts/rgb`;
                  }}
                />
              </div>

              {/* Core Quantitative Metrics - Definition List */}
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-xs pt-1">
                <div>
                  <dt className="text-text-muted text-[11px]">PNTC anomaly score</dt>
                  <dd className="font-semibold text-text-primary tabular-nums text-sm mt-0.5">
                    {formatMetric(pntc.score, 4)}
                  </dd>
                </div>
                <div>
                  <dt className="text-text-muted text-[11px]">Defects detected</dt>
                  <dd className="font-semibold text-text-primary tabular-nums text-sm mt-0.5">
                    {rep.defects?.length ?? 0}
                  </dd>
                </div>
                <div>
                  <dt className="text-text-muted text-[11px]">Max depression depth</dt>
                  <dd className="text-status-anomaly font-medium tabular-nums mt-0.5">
                    {geom.max_depression_mm !== undefined && geom.max_depression_mm !== null
                      ? `${formatMetric(geom.max_depression_mm, 2)} mm`
                      : "0.00 mm"}
                  </dd>
                </div>
                <div>
                  <dt className="text-text-muted text-[11px]">Missing material volume</dt>
                  <dd className="text-status-anomaly font-medium tabular-nums mt-0.5">
                    {vol.missing_material !== undefined
                      ? `${formatMetric(vol.missing_material, 1)} mm³`
                      : "0.0 mm³"}
                  </dd>
                </div>
                <div>
                  <dt className="text-text-muted text-[11px]">Surface area</dt>
                  <dd className="text-text-primary tabular-nums mt-0.5">
                    {morph.surface_area_3d_mm2 !== undefined
                      ? `${formatMetric(morph.surface_area_3d_mm2, 1)} mm²`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt className="text-text-muted text-[11px]">Threshold</dt>
                  <dd className="text-text-secondary tabular-nums mt-0.5">
                    {pntc.threshold || 0.5}
                  </dd>
                </div>
              </dl>

              <div className="pt-2 border-t border-white/[0.04] text-right">
                <Link
                  href={`/inspect/${detail.id}`}
                  className="inline-flex items-center gap-1.5 text-xs text-accent-primary hover:text-accent-hover font-medium transition-colors"
                >
                  <span>View full metrology report</span>
                  <ArrowRight size={13} />
                </Link>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
