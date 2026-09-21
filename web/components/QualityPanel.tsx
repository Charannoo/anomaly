"use client";

import React from "react";
import { ShieldCheck, AlertTriangle, CheckCircle2 } from "lucide-react";
import { formatMetric } from "@/lib/utils";

interface QualityPanelProps {
  report: any;
  defect: any;
}

export const QualityPanel: React.FC<QualityPanelProps> = ({ report, defect }) => {
  const qual = defect?.quality || report?.overall_quality || {};
  const guard = defect?.review_guard || report?.overall_review_guard || {};
  const reasons = guard.reasons || [];
  const isReviewRecommended = guard.status === "MANUAL_REVIEW_RECOMMENDED" || reasons.length > 0;

  return (
    <div className="space-y-4 text-xs">
      <div>
        <span className="text-xs font-mono font-semibold uppercase text-text-secondary block mb-1">
          Inspection Reliability & Quality Diagnostics
        </span>
        <p className="text-[11px] text-text-muted">
          Automated data validity checks, planar fit residuals, and human-review recommendation status.
        </p>
      </div>

      {/* Review Guard Banner */}
      <div
        className={`p-3 rounded border font-mono text-xs flex items-start gap-2.5 ${
          isReviewRecommended
            ? "border-status-review bg-status-review-bg text-status-review"
            : "border-status-normal/40 bg-status-normal-bg text-status-normal"
        }`}
      >
        {isReviewRecommended ? (
          <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
        ) : (
          <CheckCircle2 size={16} className="flex-shrink-0 mt-0.5" />
        )}
        <div>
          <div className="font-bold text-[12px]">
            {guard.recommendation || (isReviewRecommended ? "MANUAL INSPECTION RECOMMENDED" : "NOMINAL RELIABILITY")}
          </div>
          <div className="text-[11px] opacity-90 mt-1 font-sans">
            Decision Certainty: <strong>{guard.decision_certainty || "High"}</strong> · Measurement Reliability:{" "}
            <strong>{guard.measurement_reliability || "High"}</strong>
          </div>
        </div>
      </div>

      {/* Diagnostics List */}
      <div className="border border-border-subtle rounded bg-bg-app p-3 font-mono space-y-2">
        <span className="text-[10px] uppercase tracking-wider text-text-muted block">
          Sensor & Fitting Integrity
        </span>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-text-muted text-[10px] block">XYZ Point Validity</span>
            <span className="text-text-primary font-medium">
              {qual.xyz_valid_fraction !== undefined ? `${(qual.xyz_valid_fraction * 100).toFixed(1)}%` : "100.0%"}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Missing XYZ Fraction</span>
            <span className="text-text-primary font-medium">
              {qual.missing_xyz_fraction !== undefined ? `${(qual.missing_xyz_fraction * 100).toFixed(1)}%` : "0.0%"}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Surface Fit Confidence</span>
            <span className="text-status-normal font-semibold">
              {qual.surface_fit_confidence !== undefined ? `${Math.round(qual.surface_fit_confidence * 100)}%` : "85%"}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Retrieval Stability</span>
            <span className="text-text-primary uppercase font-semibold text-[11px]">
              {qual.retrieval_stability || "High"}
            </span>
          </div>
        </div>
      </div>

      {/* Warning Reasons (if any) */}
      {reasons.length > 0 && (
        <div className="border border-status-review/30 rounded bg-status-review-bg/20 p-3 space-y-1 text-xs">
          <span className="text-[11px] font-semibold text-status-review flex items-center gap-1">
            <AlertTriangle size={12} />
            Review Guard Triggers
          </span>
          <ul className="list-disc list-inside text-[11px] text-text-secondary space-y-0.5">
            {reasons.map((r: string, i: number) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default QualityPanel;
