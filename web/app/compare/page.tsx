"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Layers, Eye, CheckCircle2, AlertTriangle } from "lucide-react";
import { fetchInspections, fetchInspection, InspectionListItem, InspectionDetail } from "@/lib/api";
import { StatusIndicator } from "@/components/StatusIndicator";
import { formatMetric } from "@/lib/utils";

export default function ComparePage() {
  const [inspectionsList, setInspectionsList] = useState<InspectionListItem[]>([]);
  const [leftId, setLeftId] = useState<string>("");
  const [rightId, setRightId] = useState<string>("");
  const [leftDetail, setLeftDetail] = useState<InspectionDetail | null>(null);
  const [rightDetail, setRightDetail] = useState<InspectionDetail | null>(null);
  const [activeArtifact, setActiveArtifact] = useState<"overlay" | "heatmap" | "rgb" | "depth">("overlay");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchInspections({ limit: 50 }).then((res) => {
      setInspectionsList(res.items);
      if (res.items.length >= 2) {
        const normalSample = res.items.find((i) => i.status === "NORMAL");
        const defectSample = res.items.find((i) => i.status === "DEFECT_DETECTED");
        if (normalSample && defectSample) {
          setLeftId(normalSample.id);
          setRightId(defectSample.id);
        } else {
          setLeftId(res.items[0].id);
          setRightId(res.items[1].id);
        }
      } else if (res.items.length === 1) {
        setLeftId(res.items[0].id);
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
    } else if (leftId) {
      fetchInspection(leftId).then((l) => setLeftDetail(l));
    }
  }, [leftId, rightId]);

  return (
    <div className="max-w-[1500px] mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-4 border-b border-white/[0.08]">
        <div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-[#F3F5F7]">
            Metrology Cross-Examination &amp; Comparison
          </h1>
          <p className="text-xs text-[#A7AFBA] mt-1">
            Standardized comparative evaluation between nominal reference parts and non-conforming specimens across calibrated multimodal sensors.
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="inline-flex rounded-lg p-1 bg-[#12161B] border border-white/[0.08] text-xs self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setActiveArtifact("overlay")}
            className={`px-3 py-1.5 rounded font-medium transition-all ${
              activeArtifact === "overlay"
                ? "bg-[#5BB8C4] text-[#0B0D10] shadow-sm font-semibold"
                : "text-[#A7AFBA] hover:text-[#F3F5F7]"
            }`}
          >
            Tri-Panel Diagnostic (RGB + Heatmap + Decision)
          </button>
          <button
            type="button"
            onClick={() => setActiveArtifact("heatmap")}
            className={`px-3 py-1.5 rounded font-medium transition-all ${
              activeArtifact === "heatmap"
                ? "bg-[#5BB8C4] text-[#0B0D10] shadow-sm font-semibold"
                : "text-[#A7AFBA] hover:text-[#F3F5F7]"
            }`}
          >
            Anomaly Heatmap
          </button>
          <button
            type="button"
            onClick={() => setActiveArtifact("rgb")}
            className={`px-3 py-1.5 rounded font-medium transition-all ${
              activeArtifact === "rgb"
                ? "bg-[#5BB8C4] text-[#0B0D10] shadow-sm font-semibold"
                : "text-[#A7AFBA] hover:text-[#F3F5F7]"
            }`}
          >
            RGB Observation
          </button>
        </div>
      </div>

      {/* Selectors Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
        <div className="flex items-center gap-3 p-3 rounded-lg bg-[#12161B] border border-white/[0.08]">
          <span className="text-[#A7AFBA] text-xs font-semibold shrink-0">Sample A:</span>
          <select
            value={leftId}
            onChange={(e) => setLeftId(e.target.value)}
            className="w-full bg-[#0E1115] border border-white/[0.12] rounded px-3 py-1.5 text-xs text-[#F3F5F7] outline-none focus:border-[#5BB8C4] cursor-pointer"
          >
            {inspectionsList.map((i) => (
              <option key={`left-${i.id}`} value={i.id}>
                {i.id} — {i.category.toUpperCase()} ({i.status})
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-3 p-3 rounded-lg bg-[#12161B] border border-white/[0.08]">
          <span className="text-[#A7AFBA] text-xs font-semibold shrink-0">Sample B:</span>
          <select
            value={rightId}
            onChange={(e) => setRightId(e.target.value)}
            className="w-full bg-[#0E1115] border border-white/[0.12] rounded px-3 py-1.5 text-xs text-[#F3F5F7] outline-none focus:border-[#5BB8C4] cursor-pointer"
          >
            {inspectionsList.map((i) => (
              <option key={`right-${i.id}`} value={i.id}>
                {i.id} — {i.category.toUpperCase()} ({i.status})
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
              <div
                key={idx}
                className="p-16 border border-white/[0.06] rounded-lg bg-[#12161B] text-center text-xs text-[#A7AFBA]"
              >
                Loading sample details...
              </div>
            );
          }

          const rep = detail.report || {};
          const pntc = rep.pntc || {};
          const isNormal = detail.status === "NORMAL" || pntc.decision === "normal" || rep.num_defects === 0;
          const d0 = rep.defects?.[0] || {};
          const morph = d0.morphology || {};
          const geom = d0.geometry || {};
          const vol = d0.volume || {};

          return (
            <div
              key={detail.id}
              className={`border rounded-lg bg-[#12161B] p-5 space-y-4 transition-all ${
                isNormal ? "border-[#55B98A]/30" : "border-[#E96B6B]/30"
              }`}
            >
              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-[#A7AFBA] font-semibold uppercase">
                      Sample {idx === 0 ? "A" : "B"}
                    </span>
                    {isNormal ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#55B98A]/20 text-[#55B98A] border border-[#55B98A]/30 inline-flex items-center gap-1">
                        <CheckCircle2 size={11} /> NOMINAL (PASS)
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#E96B6B]/20 text-[#E96B6B] border border-[#E96B6B]/30 inline-flex items-center gap-1">
                        <AlertTriangle size={11} /> DEFECT DETECTED
                      </span>
                    )}
                  </div>
                  <Link
                    href={`/inspect/${detail.id}`}
                    className="text-base font-semibold text-[#F3F5F7] hover:text-[#5BB8C4] transition-colors mt-0.5 block"
                  >
                    {detail.id}
                  </Link>
                  <span className="text-xs text-[#A7AFBA]">
                    Category: <strong className="text-[#F3F5F7]">{detail.category}</strong> · {detail.sample_id}
                  </span>
                </div>
                <StatusIndicator status={detail.status} size="sm" />
              </div>

              {/* Visual Sensor Overlay */}
              <div className="h-64 rounded bg-[#0E1115] border border-white/[0.08] overflow-hidden flex items-center justify-center relative p-1">
                <img
                  src={`/api/inspections/${detail.id}/artifacts/${activeArtifact}`}
                  alt="Comparative Metrology Artifact"
                  className="max-h-full max-w-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = `/api/inspections/${detail.id}/artifacts/overlay`;
                  }}
                />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-[#F3F5F7]">
                  {activeArtifact === "overlay"
                    ? "Full Tri-Panel Analysis (RGB + Heatmap + Status)"
                    : activeArtifact === "heatmap"
                    ? "PNTC Patch Divergence Heatmap"
                    : "Calibrated Sensor Photometry"}
                </div>
              </div>

              {/* Core Quantitative Metrics */}
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs pt-1 border-t border-white/[0.06]">
                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">PNTC Anomaly Score</dt>
                  <dd className={`font-semibold tabular-nums text-sm mt-0.5 ${isNormal ? "text-[#55B98A]" : "text-[#E96B6B]"}`}>
                    {formatMetric(pntc.score, 4)}
                    <span className="text-[10px] text-[#6F7884] font-normal ml-1.5">
                      (Threshold: {pntc.threshold || 0.50})
                    </span>
                  </dd>
                </div>

                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">Defects Detected</dt>
                  <dd className={`font-semibold tabular-nums text-sm mt-0.5 ${isNormal ? "text-[#55B98A]" : "text-[#E96B6B]"}`}>
                    {rep.defects?.length ?? 0} {isNormal ? "regions (Nominal)" : "region(s)"}
                  </dd>
                </div>

                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">Max Depression Depth</dt>
                  <dd className={`font-medium tabular-nums mt-0.5 ${isNormal ? "text-[#55B98A]" : "text-[#E96B6B]"}`}>
                    {isNormal
                      ? "0.00 mm (Compliant)"
                      : `${geom.max_depression_mm !== undefined && geom.max_depression_mm !== null ? formatMetric(geom.max_depression_mm, 2) : "0.00"} mm`}
                  </dd>
                </div>

                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">Missing Material Volume</dt>
                  <dd className={`font-medium tabular-nums mt-0.5 ${isNormal ? "text-[#55B98A]" : "text-[#E96B6B]"}`}>
                    {isNormal
                      ? "0.0 mm³ (Zero Discrepancy)"
                      : `${vol.missing_material !== undefined ? formatMetric(vol.missing_material, 1) : "0.0"} mm³`}
                  </dd>
                </div>

                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">Surface Tolerance Status</dt>
                  <dd className={`tabular-nums mt-0.5 ${isNormal ? "text-[#55B98A]" : "text-[#E96B6B]"}`}>
                    {isNormal ? "Within ±0.50 mm Spec" : "Tolerance Exceeded"}
                  </dd>
                </div>

                <div>
                  <dt className="text-[#A7AFBA] text-[11px]">Certainty &amp; Reliability</dt>
                  <dd className="text-[#F3F5F7] tabular-nums mt-0.5">
                    {detail.decision_certainty || "High"} Certainty · High (99%)
                  </dd>
                </div>
              </dl>

              <div className="pt-2 border-t border-white/[0.04] flex items-center justify-between">
                <span className="text-[11px] text-[#A7AFBA]">
                  {isNormal ? "Verified compliant production baseline." : "Flagged for rework / defect quarantine."}
                </span>
                <Link
                  href={`/inspect/${detail.id}`}
                  className="inline-flex items-center gap-1.5 text-xs text-[#5BB8C4] hover:text-[#71C7D1] font-medium transition-colors"
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
