"use client";

import React from "react";
import { formatMetric } from "@/lib/utils";

interface PrototypeTraceProps {
  defect?: any;
  trace?: any;
}

export const PrototypeTrace: React.FC<PrototypeTraceProps> = ({ defect, trace: propTrace }) => {
  const trace = propTrace || defect?.prototype_trace || {};
  const rgbTopK = trace.rgb_topk || [
    { rank: 1, prototype_id: 1842, distance: 0.223, source_sample: "train/004" },
    { rank: 2, prototype_id: 2687, distance: 0.251, source_sample: "train/019" },
    { rank: 3, prototype_id: 3620, distance: 0.273, source_sample: "train/027" },
    { rank: 4, prototype_id: 9205, distance: 0.277, source_sample: "train/001" },
    { rank: 5, prototype_id: 9941, distance: 0.306, source_sample: "train/028" },
  ];

  const xyzTopK = trace.xyz_topk || [
    { rank: 1, prototype_id: 931, distance: 0.170, source_sample: "train/024" },
    { rank: 2, prototype_id: 8777, distance: 0.200, source_sample: "train/011" },
    { rank: 3, prototype_id: 1842, distance: 0.212, source_sample: "train/004" },
    { rank: 4, prototype_id: 9297, distance: 0.223, source_sample: "train/007" },
    { rank: 5, prototype_id: 13626, distance: 0.228, source_sample: "train/014" },
  ];

  // Set of shared IDs between RGB and XYZ top-k
  const rgbIds = new Set(rgbTopK.map((r: any) => r.prototype_id));
  const xyzIds = new Set(xyzTopK.map((x: any) => x.prototype_id));
  const sharedIds = new Set(Array.from(rgbIds).filter((id: any) => xyzIds.has(id)));

  const baseEvidence = trace.base_evidence ?? -0.157;
  const topologyContrib = trace.topology_contribution ?? 0.157;
  const finalEvidence = trace.final_pntc_evidence ?? 0.0;
  const jsDiv = trace.js_divergence ?? 0.87;
  const gateVal = trace.gate ?? 0.45;
  const jaccard = trace.jaccard ?? 0.111;

  return (
    <div className="space-y-4 text-xs">
      <div>
        <span className="text-xs font-mono font-semibold uppercase text-text-secondary block mb-1">
          Top-k Coreset Retrieval Trace
        </span>
        <p className="text-[11px] text-text-muted">
          Compares the nearest prototype neighborhoods retrieved by RGB tokens vs. XYZ point-cloud patches.
        </p>
      </div>

      {/* Two Columns: RGB Neighbors vs XYZ Neighbors */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        {/* RGB Column */}
        <div className="border border-border-subtle rounded bg-bg-app p-2 space-y-1.5">
          <span className="text-[10px] text-accent-primary uppercase font-semibold block">
            RGB Neighbors (k=5)
          </span>
          <div className="space-y-1">
            {rgbTopK.map((item: any) => {
              const isShared = sharedIds.has(item.prototype_id);
              return (
                <div
                  key={`rgb-${item.rank}`}
                  className={`flex items-center justify-between p-1 rounded text-[11px] ${
                    isShared ? "bg-accent-subtle border border-accent-primary/40 font-bold" : "bg-bg-subtle"
                  }`}
                >
                  <span className="text-text-muted">#{item.rank}</span>
                  <span className={isShared ? "text-accent-primary" : "text-text-primary"}>
                    P#{item.prototype_id}
                  </span>
                  <span className="text-text-muted tabular-nums">{formatMetric(item.distance, 3)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* XYZ Column */}
        <div className="border border-border-subtle rounded bg-bg-app p-2 space-y-1.5">
          <span className="text-[10px] text-accent-primary uppercase font-semibold block">
            XYZ Neighbors (k=5)
          </span>
          <div className="space-y-1">
            {xyzTopK.map((item: any) => {
              const isShared = sharedIds.has(item.prototype_id);
              return (
                <div
                  key={`xyz-${item.rank}`}
                  className={`flex items-center justify-between p-1 rounded text-[11px] ${
                    isShared ? "bg-accent-subtle border border-accent-primary/40 font-bold" : "bg-bg-subtle"
                  }`}
                >
                  <span className="text-text-muted">#{item.rank}</span>
                  <span className={isShared ? "text-accent-primary" : "text-text-primary"}>
                    P#{item.prototype_id}
                  </span>
                  <span className="text-text-muted tabular-nums">{formatMetric(item.distance, 3)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Cross-Modal Topology Divergence Metrics */}
      <div className="border border-border-subtle rounded bg-bg-subtle p-3 font-mono space-y-2">
        <span className="text-[10px] uppercase tracking-wider text-text-muted block">
          Neighborhood Topology Statistics
        </span>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-text-muted text-[10px] block">Shared Prototypes</span>
            <span className="text-text-primary font-semibold">{sharedIds.size} / 5</span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Jaccard Overlap</span>
            <span className="text-text-primary tabular-nums">{formatMetric(jaccard, 3)}</span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">JS Topology Divergence</span>
            <span className="text-status-anomaly font-semibold tabular-nums">{formatMetric(jsDiv, 3)}</span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Dynamic Gate G(p)</span>
            <span className="text-accent-primary font-semibold tabular-nums">{formatMetric(gateVal, 3)}</span>
          </div>
        </div>
      </div>

      {/* PNTC Score Decomposition Equation */}
      <div className="border border-border-subtle rounded bg-bg-app p-3 font-mono text-xs space-y-2">
        <span className="text-[10px] uppercase tracking-wider text-text-muted block">
          PNTC Score Decomposition Equation
        </span>
        <div className="flex items-center justify-between p-2 rounded bg-bg-subtle border border-border-subtle text-[11px]">
          <div className="text-center">
            <span className="text-[10px] text-text-muted block">Base Evidence</span>
            <span className="text-text-primary font-medium tabular-nums">{formatMetric(baseEvidence, 3)}</span>
          </div>
          <span className="text-text-muted font-bold">+</span>
          <div className="text-center">
            <span className="text-[10px] text-text-muted block">Topology Contrib.</span>
            <span className="text-status-anomaly font-semibold tabular-nums">+{formatMetric(topologyContrib, 3)}</span>
          </div>
          <span className="text-text-muted font-bold">=</span>
          <div className="text-center">
            <span className="text-[10px] text-text-muted block">Final Evidence</span>
            <span className="text-accent-primary font-bold tabular-nums">{formatMetric(finalEvidence, 3)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PrototypeTrace;
