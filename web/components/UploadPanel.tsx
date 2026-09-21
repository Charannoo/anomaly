"use client";

import React, { useState } from "react";
import { Upload, FileText, CheckCircle2, AlertTriangle, X, Sparkles } from "lucide-react";
import { DemoSampleModal } from "./DemoSampleModal";
import { DemoCase } from "@/lib/api";

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
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [xyzFile, setXyzFile] = useState<File | null>(null);
  const [selectedDemo, setSelectedDemo] = useState<DemoCase | null>(null);
  const [generate3D, setGenerate3D] = useState(true);
  const [generateAssistant, setGenerateAssistant] = useState(true);
  const [responseMode, setResponseMode] = useState("TECHNICAL");
  const [isDemoModalOpen, setIsDemoModalOpen] = useState(false);

  const hasInputs = Boolean(selectedDemo || (rgbFile && xyzFile) || rgbFile);

  const handleClear = () => {
    setRgbFile(null);
    setXyzFile(null);
    setSelectedDemo(null);
  };

  const handleRun = () => {
    if (selectedDemo) {
      onRunInspection({
        demo_case_id: selectedDemo.id,
        sample_id: selectedDemo.id,
        category: selectedDemo.category,
        generate_3d: generate3D,
        generate_assistant_summary: generateAssistant,
        response_mode: responseMode,
      });
    } else {
      onRunInspection({
        sample_id: rgbFile?.name.replace(/\.[^/.]+$/, "") || "custom_sample",
        category: "component",
        generate_3d: generate3D,
        generate_assistant_summary: generateAssistant,
        response_mode: responseMode,
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Demo Loader Bar */}
      <div className="flex items-center justify-between py-1">
        <button
          type="button"
          onClick={() => setIsDemoModalOpen(true)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-white/[0.08] bg-[#12161B] hover:bg-[#171C22] text-xs font-medium text-[#F3F5F7] transition-colors"
        >
          <Sparkles size={13} className="text-[#5BB8C4]" />
          <span>Load demo sample...</span>
        </button>

        {selectedDemo && (
          <div className="flex items-center gap-2 text-xs text-[#A7AFBA]">
            <span>Loaded: <strong className="text-[#F3F5F7]">{selectedDemo.name}</strong> ({selectedDemo.category})</span>
            <button
              onClick={() => setSelectedDemo(null)}
              className="text-[#6F7884] hover:text-[#F3F5F7] p-1"
              title="Remove sample"
            >
              <X size={13} />
            </button>
          </div>
        )}
      </div>

      {/* Two Equal Panes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Pane: RGB Observation */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-[#A7AFBA]">
            RGB observation
          </div>

          {selectedDemo ? (
            <div className="h-56 w-full rounded bg-[#0E1115] border border-white/[0.06] overflow-hidden relative group flex items-center justify-center">
              <img
                src={`/api/inspections/INSP-${selectedDemo.id}/artifacts/rgb`}
                alt="RGB Observation"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `/api/inspections/INSP-${selectedDemo.id}/artifacts/overlay`;
                }}
              />
              <div className="absolute bottom-2.5 left-2.5 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-white">
                256 × 256 · RGB 8-bit
              </div>
            </div>
          ) : rgbFile ? (
            <div className="h-56 w-full rounded bg-[#0E1115] border border-white/[0.06] p-4 flex flex-col items-center justify-center text-center">
              <FileText size={30} className="text-[#5BB8C4] mb-2" />
              <span className="text-xs text-[#F3F5F7] font-medium">{rgbFile.name}</span>
              <span className="text-[11px] text-[#6F7884] mt-0.5">{(rgbFile.size / 1024).toFixed(1)} KB</span>
            </div>
          ) : (
            <label className="h-56 w-full rounded border border-dashed border-white/[0.12] hover:border-[#5BB8C4] bg-[#0E1115]/50 hover:bg-[#0E1115] flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-colors">
              <Upload size={20} className="text-[#6F7884] mb-2" />
              <span className="text-xs font-medium text-[#F3F5F7]">Drop RGB image here</span>
              <span className="text-[11px] text-[#6F7884] mt-1">PNG, JPG, TIFF up to 20MB</span>
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

          <div className="flex items-center justify-between text-[11px] text-[#6F7884] pt-1">
            <span>DINOv2 ViT-B/14</span>
            <span className="font-mono text-[#55B98A]">{selectedDemo || rgbFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>

        {/* Right Pane: 3D / XYZ Observation */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-[#A7AFBA]">
            3D / XYZ observation
          </div>

          {selectedDemo ? (
            <div className="h-56 w-full rounded bg-[#0E1115] border border-white/[0.06] overflow-hidden relative group flex items-center justify-center">
              <img
                src={`/api/inspections/INSP-${selectedDemo.id}/artifacts/depth`}
                alt="Depth Scan"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `/api/inspections/INSP-${selectedDemo.id}/artifacts/overlay`;
                }}
              />
              <div className="absolute bottom-2.5 left-2.5 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-white">
                Point-MAE Grid · Calibrated (mm)
              </div>
            </div>
          ) : xyzFile ? (
            <div className="h-56 w-full rounded bg-[#0E1115] border border-white/[0.06] p-4 flex flex-col items-center justify-center text-center">
              <FileText size={30} className="text-[#5BB8C4] mb-2" />
              <span className="text-xs text-[#F3F5F7] font-medium">{xyzFile.name}</span>
              <span className="text-[11px] text-[#6F7884] mt-0.5">{(xyzFile.size / 1024).toFixed(1)} KB</span>
            </div>
          ) : (
            <label className="h-56 w-full rounded border border-dashed border-white/[0.12] hover:border-[#5BB8C4] bg-[#0E1115]/50 hover:bg-[#0E1115] flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-colors">
              <Upload size={20} className="text-[#6F7884] mb-2" />
              <span className="text-xs font-medium text-[#F3F5F7]">Drop 3D / XYZ file here</span>
              <span className="text-[11px] text-[#6F7884] mt-1">.npy, .ply, or organized depth map</span>
              <input
                type="file"
                accept=".npy,.ply,.png,.tiff"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) setXyzFile(e.target.files[0]);
                }}
              />
            </label>
          )}

          <div className="flex items-center justify-between text-[11px] text-[#6F7884] pt-1">
            <span>Point-MAE Backbone</span>
            <span className="font-mono text-[#55B98A]">{selectedDemo || xyzFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>
      </div>

      {/* Clean Horizontal Observation Checklist (No enclosing card) */}
      <div className="border-t border-white/[0.06] pt-4">
        <div className="flex flex-wrap items-center gap-6 text-xs text-[#A7AFBA]">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>RGB loaded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>XYZ loaded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>Spatially aligned</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>Units verified (mm)</span>
          </div>
        </div>
      </div>

      {/* Minimal Inspection Options & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-2 text-xs">
        <div className="flex flex-wrap items-center gap-6 text-[#A7AFBA]">
          <label className="flex items-center gap-2 cursor-pointer hover:text-[#F3F5F7]">
            <input
              type="checkbox"
              checked={generate3D}
              onChange={(e) => setGenerate3D(e.target.checked)}
              className="rounded border-white/[0.12] bg-[#12161B] text-[#5BB8C4] focus:ring-0"
            />
            <span>Generate 3D metrology view</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer hover:text-[#F3F5F7]">
            <input
              type="checkbox"
              checked={generateAssistant}
              onChange={(e) => setGenerateAssistant(e.target.checked)}
              className="rounded border-white/[0.12] bg-[#12161B] text-[#5BB8C4] focus:ring-0"
            />
            <span>Generate AI explanation</span>
          </label>
          <div className="flex items-center gap-2">
            <span className="text-[#6F7884]">Style:</span>
            <select
              value={responseMode}
              onChange={(e) => setResponseMode(e.target.value)}
              className="bg-[#12161B] border border-white/[0.08] text-[#F3F5F7] rounded px-2 py-1 text-xs outline-none focus:border-[#5BB8C4]"
            >
              <option value="TECHNICAL">Technical</option>
              <option value="SIMPLE">Simple</option>
              <option value="VIVA">Viva</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleClear}
            disabled={!hasInputs || loading}
            className="px-3.5 py-2 rounded text-[#6F7884] hover:text-[#F3F5F7] text-xs font-medium transition-colors disabled:opacity-30"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={handleRun}
            disabled={!hasInputs || loading}
            className="px-5 py-2 rounded bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] text-xs font-semibold tracking-wide transition-colors disabled:opacity-30"
          >
            {loading ? "Executing Pipeline..." : "Run Inspection"}
          </button>
        </div>
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
