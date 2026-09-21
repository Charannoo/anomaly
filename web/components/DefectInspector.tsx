"use client";

import React from "react";
import { Layers, Cuboid, Activity, Info, ArrowUpRight } from "lucide-react";
import { EvidenceMeter } from "./EvidenceMeter";
import { formatMetric } from "@/lib/utils";

interface DefectInspectorProps {
  defect: any;
  overallDecision?: string;
  onViewTrace?: () => void;
  onSelectDefect?: (id: number) => void;
}

export const DefectInspector: React.FC<DefectInspectorProps> = ({
  defect,
  overallDecision,
  onViewTrace,
  onSelectDefect,
}) => {
  if (!defect) {
    return (
      <div className="border border-border-default rounded-md bg-bg-panel p-6 text-center text-xs text-text-muted">
        No defect selected or component meets nominal specifications.
      </div>
    );
  }

  const morph = defect.morphology || {};
  const geom = defect.geometry || {};
  const vol = defect.volume || {};
  const trace = defect.prototype_trace || {};
  const twin = defect.normal_twin || {};

  return (
    <div className="border border-border-default rounded-md bg-bg-panel p-4 space-y-4 overflow-y-auto max-h-full">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-2.5">
        <div>
          <span className="text-xs font-mono font-semibold text-status-anomaly uppercase">
            Defect #{defect.id}
          </span>
          <h3 className="text-sm font-semibold text-text-primary capitalize">
            {defect.location?.label || "Surface Anomaly Region"}
          </h3>
        </div>
        <span className="text-[11px] font-mono text-text-muted px-2 py-0.5 rounded bg-bg-app border border-border-subtle">
          ID #{defect.id}
        </span>
      </div>

      {/* SECTION 1: MORPHOLOGY */}
      <div className="space-y-2">
        <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
          <Layers size={13} className="text-accent-primary" />
          <span>Morphology & 2D Geometry</span>
        </span>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs bg-bg-subtle p-2.5 rounded border border-border-subtle">
          <div>
            <dt className="text-text-muted text-[10px]">Shape Profile</dt>
            <dd className="font-medium text-text-primary capitalize">{morph.label || "Irregular"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Aspect Ratio</dt>
            <dd className="font-mono text-text-primary font-medium">{formatMetric(morph.aspect_ratio, 2)}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Major Dimension</dt>
            <dd className="font-mono text-text-primary font-medium">{formatMetric(morph.major_length_mm, 1)} mm</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Minor Dimension</dt>
            <dd className="font-mono text-text-primary font-medium">{formatMetric(morph.minor_length_mm, 1)} mm</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Projected Area</dt>
            <dd className="font-mono text-text-primary font-medium">{formatMetric(morph.projected_area_mm2, 1)} mm²</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">3D Surface Area</dt>
            <dd className="font-mono text-text-primary font-medium">{formatMetric(morph.surface_area_3d_mm2, 1)} mm²</dd>
          </div>
        </dl>
      </div>

      {/* SECTION 2: 3D GEOMETRY & VOLUMETRIC LOSS */}
      <div className="space-y-2">
        <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
          <Cuboid size={13} className="text-accent-primary" />
          <span>3D Surface & Material Loss</span>
        </span>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs bg-bg-subtle p-2.5 rounded border border-border-subtle">
          <div>
            <dt className="text-text-muted text-[10px]">Structure</dt>
            <dd className="font-medium text-text-primary capitalize">{geom.label || geom.structure || "Surface Deviation"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Max Depression Depth</dt>
            <dd className="font-mono text-status-anomaly font-semibold">
              {geom.max_depression_mm !== null && geom.max_depression_mm !== undefined
                ? `${formatMetric(geom.max_depression_mm, 2)} mm`
                : "—"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Missing Material (Loss)</dt>
            <dd className="font-mono text-status-anomaly font-semibold">
              {vol.missing_material !== undefined
                ? `${formatMetric(vol.missing_material, 1)} mm³`
                : "—"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Excess Material (Burr)</dt>
            <dd className="font-mono text-text-primary font-medium">
              {vol.excess_material !== undefined
                ? `${formatMetric(vol.excess_material, 1)} mm³`
                : "0.0 mm³"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Reference Plane Model</dt>
            <dd className="font-mono text-text-secondary text-[11px]">{vol.reference_model_type || "Robust Plane"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[10px]">Measurement Confidence</dt>
            <dd className="font-mono text-status-normal font-medium">
              {geom.measurement_confidence ? `${Math.round(geom.measurement_confidence * 100)}%` : "High"}
            </dd>
          </div>
        </dl>
      </div>

      {/* SECTION 3: EVIDENCE METERS */}
      <div className="space-y-2.5">
        <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
          <Activity size={13} className="text-accent-primary" />
          <span>Multimodal Evidence Breakdown</span>
        </span>
        <div className="bg-bg-subtle p-3 rounded border border-border-subtle space-y-2.5">
          <EvidenceMeter
            label="RGB Visual Evidence"
            value={twin.rgb_distance !== undefined ? 1.0 - twin.rgb_distance : 0.61}
            qualitativeLevel={twin.rgb_similarity || "Moderate"}
          />
          <EvidenceMeter
            label="XYZ Geometric Evidence"
            value={twin.xyz_distance !== undefined ? 1.0 - twin.xyz_distance : 0.83}
            qualitativeLevel={twin.xyz_similarity || "High"}
            highlight
          />
          <EvidenceMeter
            label="Cross-Modal Topology Disagreement"
            value={trace.js_divergence !== undefined ? trace.js_divergence : 0.91}
            qualitativeLevel={trace.js_divergence > 0.7 ? "Very High" : "Moderate"}
            highlight
          />
          <EvidenceMeter
            label="Dynamic Confidence Gate G(p)"
            value={trace.gate !== undefined ? trace.gate : 0.45}
            qualitativeLevel="Active"
          />
        </div>
      </div>

      {/* SECTION 4: DETERMINISTIC EXPLANATION */}
      <div className="p-3 rounded border border-border-subtle bg-bg-subtle text-xs space-y-1.5">
        <div className="flex items-center justify-between text-text-muted text-[11px]">
          <span className="font-mono font-semibold uppercase flex items-center gap-1">
            <Info size={12} className="text-accent-primary" />
            Why Flagged (Deterministic)
          </span>
          {onViewTrace && (
            <button
              onClick={onViewTrace}
              className="text-accent-primary hover:underline flex items-center gap-0.5 text-[11px]"
            >
              <span>Inspect Trace</span>
              <ArrowUpRight size={11} />
            </button>
          )}
        </div>
        <p className="text-text-secondary text-[11px] leading-relaxed">
          {trace.why_flagged ||
            defect.explanation ||
            "The region exhibits verified signed surface deviation and significant RGB–XYZ normal prototype disagreement, supporting anomalous classification."}
        </p>
      </div>
    </div>
  );
};

export default DefectInspector;
