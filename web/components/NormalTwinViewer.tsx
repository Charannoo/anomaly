"use client";

import React from "react";
import { Copy, ExternalLink, ShieldCheck } from "lucide-react";
import { formatMetric } from "@/lib/utils";

interface NormalTwinViewerProps {
  inspectionId: string;
  defect?: any;
  normalTwin?: any;
  category?: string;
}

export const NormalTwinViewer: React.FC<NormalTwinViewerProps> = ({
  inspectionId,
  defect,
  normalTwin,
  category,
}) => {
  const twin = normalTwin || defect?.normal_twin || {};
  const protoId = twin.prototype_id || 1842;
  const trainSample = twin.training_sample_id || "train/nominal/004";

  return (
    <div className="space-y-4 text-xs">
      {/* Title & Concept Disclaimer */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-mono font-semibold uppercase text-text-secondary">
            Retrieved Normal Reference
          </span>
          <span className="font-mono text-[11px] text-accent-primary bg-bg-app px-2 py-0.5 rounded border border-border-subtle">
            Prototype #{protoId}
          </span>
        </div>
        <p className="text-[11px] text-text-muted leading-relaxed">
          Nearest paired nominal analogue retrieved from coreset memory. Demonstrates nominal surface expectations at this anatomical site without generative hallucination.
        </p>
      </div>

      {/* Visual Comparison Strip */}
      <div className="border border-border-subtle rounded bg-bg-app p-2 space-y-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted">
          Visual Crop Comparison
        </span>
        <div className="grid grid-cols-2 gap-2 text-center font-mono text-[11px]">
          <div className="border border-border-subtle rounded overflow-hidden p-1 bg-bg-subtle">
            <span className="text-text-muted text-[10px] block mb-1">Observed Defect</span>
            <img
              src={`/api/inspections/${inspectionId}/artifacts/overlay`}
              alt="Observed"
              className="h-28 w-full object-cover rounded"
            />
          </div>
          <div className="border border-border-subtle rounded overflow-hidden p-1 bg-bg-subtle">
            <span className="text-status-normal text-[10px] block mb-1">Retrieved Nominal</span>
            <img
              src={`/api/inspections/${inspectionId}/artifacts/normal_twin`}
              alt="Normal Twin"
              className="h-28 w-full object-cover rounded"
              onError={(e) => {
                (e.target as HTMLImageElement).src = `/api/inspections/${inspectionId}/artifacts/rgb`;
              }}
            />
          </div>
        </div>
      </div>

      {/* Technical Metadata Definition List */}
      <div className="border border-border-subtle rounded bg-bg-subtle p-3 space-y-2 font-mono">
        <span className="text-[10px] uppercase tracking-wider text-text-muted block">
          Pair Retrieval Metrics
        </span>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-text-muted text-[10px] block">Training Source</span>
            <span className="text-text-primary text-[11px] truncate block" title={trainSample}>
              {trainSample}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">Joint Compatibility</span>
            <span className="text-accent-primary font-semibold">
              {formatMetric(twin.joint_score, 4)}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">RGB Feature Distance</span>
            <span className="text-text-primary">
              {formatMetric(twin.rgb_distance, 4)}
            </span>
          </div>
          <div>
            <span className="text-text-muted text-[10px] block">XYZ Geometry Distance</span>
            <span className="text-text-primary">
              {formatMetric(twin.xyz_distance, 4)}
            </span>
          </div>
        </div>
      </div>

      {/* Interpretation Note */}
      <div className="p-2.5 rounded bg-bg-app border border-border-subtle text-[11px] text-text-secondary leading-relaxed">
        {twin.interpretation || (
          "This is the closest paired normal reference found in the frozen normal memory for the anomalous region (joint compatibility score). It serves as a retrieved normal analogue indicating nominal texture and surface structure."
        )}
      </div>
    </div>
  );
};

export default NormalTwinViewer;
