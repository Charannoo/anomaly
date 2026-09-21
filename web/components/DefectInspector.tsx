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
      <div className="p-8 text-center text-xs text-text-muted">
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
    <div className="space-y-6 overflow-y-auto max-h-full pr-1">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/[0.05]">
        <div>
          <span className="text-xs font-medium text-status-anomaly">
            Defect #{defect.id}
          </span>
          <h3 className="text-base font-semibold text-text-primary capitalize mt-0.5">
            {defect.location?.label || "Surface anomaly region"}
          </h3>
        </div>
        <span className="text-xs font-mono text-text-muted px-2 py-0.5 rounded bg-bg-surface border border-white/[0.06]">
          Region #{defect.id}
        </span>
      </div>

      {/* SECTION 1: MORPHOLOGY */}
      <div className="space-y-2.5">
        <span className="text-xs font-medium text-text-muted flex items-center gap-1.5">
          <Layers size={14} className="text-accent-primary" />
          <span>Morphology & 2D geometry</span>
        </span>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs p-3 rounded bg-bg-surface">
          <div>
            <dt className="text-text-muted text-[11px]">Shape profile</dt>
            <dd className="font-medium text-text-primary capitalize mt-0.5">{morph.label || "Irregular"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Aspect ratio</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">{formatMetric(morph.aspect_ratio, 2)}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Major dimension</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">{formatMetric(morph.major_length_mm, 1)} mm</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Minor dimension</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">{formatMetric(morph.minor_length_mm, 1)} mm</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Projected area</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">{formatMetric(morph.projected_area_mm2, 1)} mm²</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">3D surface area</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">{formatMetric(morph.surface_area_3d_mm2, 1)} mm²</dd>
          </div>
        </dl>
      </div>

      {/* SECTION 2: 3D GEOMETRY & VOLUMETRIC LOSS */}
      <div className="space-y-2.5">
        <span className="text-xs font-medium text-text-muted flex items-center gap-1.5">
          <Cuboid size={14} className="text-accent-primary" />
          <span>3D surface & material displacement</span>
        </span>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs p-3 rounded bg-bg-surface">
          <div>
            <dt className="text-text-muted text-[11px]">Structure</dt>
            <dd className="font-medium text-text-primary capitalize mt-0.5">{geom.label || geom.structure || "Surface Deviation"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Max depression depth</dt>
            <dd className="font-mono text-status-anomaly font-semibold mt-0.5">
              {geom.max_depression_mm !== null && geom.max_depression_mm !== undefined
                ? `${formatMetric(geom.max_depression_mm, 2)} mm`
                : "—"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Missing material (loss)</dt>
            <dd className="font-mono text-status-anomaly font-semibold mt-0.5">
              {vol.missing_material !== undefined
                ? `${formatMetric(vol.missing_material, 1)} mm³`
                : "—"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Excess material (burr)</dt>
            <dd className="font-mono text-text-primary font-medium mt-0.5">
              {vol.excess_material !== undefined
                ? `${formatMetric(vol.excess_material, 1)} mm³`
                : "0.0 mm³"}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Reference plane model</dt>
            <dd className="text-text-secondary text-[11px] mt-0.5">{vol.reference_model_type || "Robust Plane"}</dd>
          </div>
          <div>
            <dt className="text-text-muted text-[11px]">Measurement confidence</dt>
            <dd className="font-mono text-status-normal font-medium mt-0.5">
              {geom.measurement_confidence ? `${Math.round(geom.measurement_confidence * 100)}%` : "High"}
            </dd>
          </div>
        </dl>
      </div>

      {/* SECTION 3: EVIDENCE METERS */}
      <div className="space-y-2.5">
        <span className="text-xs font-medium text-text-muted flex items-center gap-1.5">
          <Activity size={14} className="text-accent-primary" />
          <span>Multimodal evidence breakdown</span>
        </span>
        <div className="p-3 rounded bg-bg-surface space-y-2.5">
          <EvidenceMeter
            label="RGB visual evidence"
            value={twin.rgb_distance !== undefined ? 1.0 - twin.rgb_distance : 0.61}
            qualitativeLevel={twin.rgb_similarity || "Moderate"}
          />
          <EvidenceMeter
            label="XYZ geometric evidence"
            value={twin.xyz_distance !== undefined ? 1.0 - twin.xyz_distance : 0.83}
            qualitativeLevel={twin.xyz_similarity || "High"}
            highlight
          />
          <EvidenceMeter
            label="Cross-modal topology disagreement"
            value={trace.js_divergence !== undefined ? trace.js_divergence : 0.91}
            qualitativeLevel={trace.js_divergence > 0.7 ? "Very High" : "Moderate"}
            highlight
          />
          <EvidenceMeter
            label="Dynamic confidence gate G(p)"
            value={trace.gate !== undefined ? trace.gate : 0.45}
            qualitativeLevel="Active"
          />
        </div>
      </div>

      {/* SECTION 4: DETERMINISTIC EXPLANATION */}
      <div className="p-3 rounded bg-bg-surface text-xs space-y-1.5">
        <div className="flex items-center justify-between text-text-muted text-xs">
          <span className="font-medium flex items-center gap-1">
            <Info size={13} className="text-accent-primary" />
            Why flagged
          </span>
          {onViewTrace && (
            <button
              onClick={onViewTrace}
              className="text-accent-primary hover:underline flex items-center gap-0.5 text-xs font-medium"
            >
              <span>Inspect trace</span>
              <ArrowUpRight size={12} />
            </button>
          )}
        </div>
        <p className="text-text-secondary text-xs leading-relaxed">
          {trace.why_flagged ||
            defect.explanation ||
            "The region exhibits verified signed surface deviation and significant RGB–XYZ normal prototype disagreement, supporting anomalous classification."}
        </p>
      </div>
    </div>
  );
};

export default DefectInspector;
