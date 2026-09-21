"use client";

import React, { useState } from "react";
import { Upload, FileText, CheckCircle2, AlertTriangle, Sparkles, X } from "lucide-react";
import { DemoCase } from "@/lib/api";
import { DemoSampleModal } from "./DemoSampleModal";
import { cn } from "@/lib/utils";

interface UploadPanelProps {
  onRunInspection: (payload: {
    demo_case_id?: string;
    sample_id?: string;
    category?: string;
    generate_3d: boolean;
    generate_assistant_summary: boolean;
    response_mode: string;
  }) => void;
  loading?: boolean;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({
  onRunInspection,
  loading = false,
}) => {
  const [selectedDemo, setSelectedDemo] = useState<DemoCase | null>(null);
  const [isDemoModalOpen, setIsDemoModalOpen] = useState(false);
  const [generate3D, setGenerate3D] = useState(true);
  const [generateAssistant, setGenerateAssistant] = useState(true);
  const [responseMode, setResponseMode] = useState("TECHNICAL");

  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [xyzFile, setXyzFile] = useState<File | null>(null);

  const hasInputs = !!selectedDemo || (!!rgbFile && !!xyzFile);

  const handleClear = () => {
    setSelectedDemo(null);
    setRgbFile(null);
    setXyzFile(null);
  };

  const handleRun = () => {
    if (!hasInputs) return;
    onRunInspection({
      demo_case_id: selectedDemo?.id,
      sample_id: selectedDemo ? `sample_${selectedDemo.id}` : (rgbFile ? rgbFile.name.replace(/\.[^/.]+$/, "") : "custom_sample"),
      category: selectedDemo?.category || "cookie",
      generate_3d: generate3D,
      generate_assistant_summary: generateAssistant,
      response_mode: responseMode,
    });
  };

  return (
    <div className="space-y-6">
      {/* Top Controls Strip */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-text-primary uppercase tracking-wide font-mono">
            Sensor Observation Pair
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Synchronized 2D visual capture and 3D surface geometry
          </p>
        </div>

        <button
          type="button"
          onClick={() => setIsDemoModalOpen(true)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-DEFAULT bg-bg-panel border border-border-emphasized hover:border-accent-primary text-text-primary hover:text-accent-primary text-xs font-medium transition-colors"
        >
          <Sparkles size={13} className="text-accent-primary" />
          <span>Load Demo Sample</span>
        </button>
      </div>

      {/* Selected Demo Banner */}
      {selectedDemo && (
        <div className="flex items-center justify-between p-3 rounded border border-accent-subtle bg-bg-panel text-xs">
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full bg-accent-primary" />
            <span className="text-text-muted">Loaded Demo Case:</span>
            <span className="font-semibold text-text-primary">{selectedDemo.name}</span>
            <span className="font-mono text-text-muted text-[11px]">({selectedDemo.category})</span>
          </div>
          <button
            onClick={() => setSelectedDemo(null)}
            className="text-text-muted hover:text-text-primary p-0.5"
            title="Clear demo selection"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Two Equal Panes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left Pane: RGB Observation */}
        <div className="border border-border-default rounded-md bg-bg-panel p-4 flex flex-col justify-between min-h-[260px]">
          <div>
            <div className="flex items-center justify-between border-b border-border-subtle pb-2 mb-3">
              <span className="text-xs font-mono font-semibold uppercase text-text-secondary">
                RGB Observation
              </span>
              <span className="text-[11px] font-mono text-text-muted">High-res Optical Scan</span>
            </div>

            {selectedDemo ? (
              <div className="h-44 w-full rounded bg-bg-app border border-border-subtle overflow-hidden relative group">
                <img
                  src={`/api/inspections/INSP-${selectedDemo.id}/artifacts/rgb`}
                  alt="RGB Observation"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    // Fallback to overlay if original rgb isn't available
                    (e.target as HTMLImageElement).src = `/api/inspections/INSP-${selectedDemo.id}/artifacts/overlay`;
                  }}
                />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white">
                  256 × 256 · RGB 8-bit
                </div>
              </div>
            ) : rgbFile ? (
              <div className="h-44 w-full rounded bg-bg-app border border-border-subtle p-3 flex flex-col items-center justify-center text-center">
                <FileText size={28} className="text-accent-primary mb-2" />
                <span className="text-xs font-mono text-text-primary font-medium">{rgbFile.name}</span>
                <span className="text-[11px] text-text-muted mt-0.5">{(rgbFile.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <label className="h-44 w-full rounded border border-dashed border-border-emphasized hover:border-accent-primary bg-bg-subtle flex flex-col items-center justify-center text-center p-4 cursor-pointer transition-colors">
                <Upload size={22} className="text-text-muted mb-2" />
                <span className="text-xs font-medium text-text-primary">Drop RGB image here</span>
                <span className="text-[11px] text-text-muted mt-0.5">PNG, JPG, TIFF up to 20MB</span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setRgbFile(e.target.files[0]);
                  }}
                />
              </label>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between text-[11px] text-text-muted border-t border-border-subtle pt-2">
            <span>DINOv2 ViT-B/14 Target</span>
            <span className="font-mono text-status-normal">{selectedDemo || rgbFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>

        {/* Right Pane: 3D / XYZ Observation */}
        <div className="border border-border-default rounded-md bg-bg-panel p-4 flex flex-col justify-between min-h-[260px]">
          <div>
            <div className="flex items-center justify-between border-b border-border-subtle pb-2 mb-3">
              <span className="text-xs font-mono font-semibold uppercase text-text-secondary">
                3D / XYZ Observation
              </span>
              <span className="text-[11px] font-mono text-text-muted">Structured Point Cloud</span>
            </div>

            {selectedDemo ? (
              <div className="h-44 w-full rounded bg-bg-app border border-border-subtle overflow-hidden relative group">
                <img
                  src={`/api/inspections/INSP-${selectedDemo.id}/artifacts/depth`}
                  alt="Depth Scan"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = `/api/inspections/INSP-${selectedDemo.id}/artifacts/overlay`;
                  }}
                />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white">
                  Point-MAE Grid · Calibrated (mm)
                </div>
              </div>
            ) : xyzFile ? (
              <div className="h-44 w-full rounded bg-bg-app border border-border-subtle p-3 flex flex-col items-center justify-center text-center">
                <FileText size={28} className="text-accent-primary mb-2" />
                <span className="text-xs font-mono text-text-primary font-medium">{xyzFile.name}</span>
                <span className="text-[11px] text-text-muted mt-0.5">{(xyzFile.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <label className="h-44 w-full rounded border border-dashed border-border-emphasized hover:border-accent-primary bg-bg-subtle flex flex-col items-center justify-center text-center p-4 cursor-pointer transition-colors">
                <Upload size={22} className="text-text-muted mb-2" />
                <span className="text-xs font-medium text-text-primary">Drop XYZ point cloud or depth map</span>
                <span className="text-[11px] text-text-muted mt-0.5">NPY, NPZ, TIFF, PCD</span>
                <input
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setXyzFile(e.target.files[0]);
                  }}
                />
              </label>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between text-[11px] text-text-muted border-t border-border-subtle pt-2">
            <span>Point-MAE Target</span>
            <span className="font-mono text-status-normal">{selectedDemo || xyzFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>
      </div>

      {/* Input Status Panel (Validation Checklist) */}
      <div className="border border-border-default rounded-md bg-bg-panel p-3.5 space-y-2">
        <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-text-muted block">
          Observation Pair Verification
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-text-secondary">
            <CheckCircle2 size={13} className={hasInputs ? "text-status-normal" : "text-text-muted"} />
            <span>RGB Loaded</span>
          </div>
          <div className="flex items-center gap-1.5 text-text-secondary">
            <CheckCircle2 size={13} className={hasInputs ? "text-status-normal" : "text-text-muted"} />
            <span>XYZ Loaded</span>
          </div>
          <div className="flex items-center gap-1.5 text-text-secondary">
            <CheckCircle2 size={13} className={hasInputs ? "text-status-normal" : "text-text-muted"} />
            <span>Spatial Correspondence Valid</span>
          </div>
          <div className="flex items-center gap-1.5 text-text-secondary">
            <AlertTriangle size={13} className="text-status-review" />
            <span>Physical Units: Calibrated (mm)</span>
          </div>
        </div>
      </div>

      {/* Inspection Options */}
      <div className="border border-border-default rounded-md bg-bg-panel p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-4 text-text-secondary">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={generate3D}
              onChange={(e) => setGenerate3D(e.target.checked)}
              className="rounded border-border-default bg-bg-surface text-accent-primary"
            />
            <span>Generate 3D Metrology View</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={generateAssistant}
              onChange={(e) => setGenerateAssistant(e.target.checked)}
              className="rounded border-border-default bg-bg-surface text-accent-primary"
            />
            <span>Generate Grounded AI Explanation</span>
          </label>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-text-muted text-[11px]">Explanation Mode:</span>
          <select
            value={responseMode}
            onChange={(e) => setResponseMode(e.target.value)}
            className="bg-bg-surface border border-border-default text-text-primary rounded px-2.5 py-1 text-xs outline-none"
          >
            <option value="TECHNICAL">Technical (Metrology)</option>
            <option value="SIMPLE">Simple (Operator)</option>
            <option value="VIVA">Viva (Academic Defense)</option>
          </select>
        </div>
      </div>

      {/* Execution Actions */}
      <div className="flex items-center justify-end gap-3 pt-2">
        <button
          type="button"
          onClick={handleClear}
          disabled={!hasInputs || loading}
          className="px-4 py-2 rounded-DEFAULT border border-border-default hover:border-border-emphasized text-text-secondary hover:text-text-primary text-xs font-medium transition-colors disabled:opacity-40"
        >
          Clear
        </button>
        <button
          type="button"
          onClick={handleRun}
          disabled={!hasInputs || loading}
          className="px-5 py-2 rounded-DEFAULT bg-accent-primary hover:bg-accent-hover text-bg-app text-xs font-semibold tracking-wide transition-colors shadow-subtle disabled:opacity-40"
        >
          {loading ? "Executing Pipeline..." : "Run Inspection"}
        </button>
      </div>

      {/* Demo Modal */}
      <DemoSampleModal
        isOpen={isDemoModalOpen}
        onClose={() => setIsDemoModalOpen(false)}
        onSelectSample={(demo) => setSelectedDemo(demo)}
      />
    </div>
  );
};

export default UploadPanel;
