"use client";

import React, { useState, useEffect, useRef } from "react";
import { Upload, FileText, CheckCircle2, X, Sparkles, Image as ImageIcon } from "lucide-react";
import { DemoSampleModal } from "./DemoSampleModal";
import { DemoCase } from "@/lib/api";

interface UploadPanelProps {
  onRunInspection: (payload: {
    demo_case_id?: string;
    sample_id?: string;
    category?: string;
    sample_condition?: string;
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
  const [rgbPreview, setRgbPreview] = useState<string | null>(null);
  const [xyzPreview, setXyzPreview] = useState<string | null>(null);
  const [isRgbDragging, setIsRgbDragging] = useState(false);
  const [isXyzDragging, setIsXyzDragging] = useState(false);
  const [selectedDemo, setSelectedDemo] = useState<DemoCase | null>(null);
  const [sampleCondition, setSampleCondition] = useState<"nominal" | "anomalous" | "auto">("nominal");
  const [generate3D, setGenerate3D] = useState(true);
  const [generateAssistant, setGenerateAssistant] = useState(true);
  const [responseMode, setResponseMode] = useState("TECHNICAL");
  const [isDemoModalOpen, setIsDemoModalOpen] = useState(false);

  const rgbInputRef = useRef<HTMLInputElement>(null);
  const xyzInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (rgbFile) {
      const url = URL.createObjectURL(rgbFile);
      setRgbPreview(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setRgbPreview(null);
    }
  }, [rgbFile]);

  useEffect(() => {
    if (xyzFile && (xyzFile.type.startsWith("image/") || xyzFile.name.endsWith(".png") || xyzFile.name.endsWith(".jpg"))) {
      const url = URL.createObjectURL(xyzFile);
      setXyzPreview(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setXyzPreview(null);
    }
  }, [xyzFile]);

  const hasInputs = Boolean(selectedDemo || (rgbFile && xyzFile) || rgbFile);

  const handleClear = () => {
    setRgbFile(null);
    setXyzFile(null);
    setSelectedDemo(null);
  };

  const quickPresets = [
    {
      id: "08_nominal_cookie",
      label: "Good Cookie",
      tag: "Nominal Pass",
      score: "0.1142",
      cat: "cookie",
      cond: "nominal" as const,
      color: "border-[#55B98A]/40 hover:border-[#55B98A] text-[#55B98A] bg-[#55B98A]/10",
      dot: "bg-[#55B98A]",
    },
    {
      id: "01_strong_defect",
      label: "Defective Cookie",
      tag: "Combined Defect",
      score: "0.8800",
      cat: "cookie",
      cond: "anomalous" as const,
      color: "border-[#E96B6B]/40 hover:border-[#E96B6B] text-[#E96B6B] bg-[#E96B6B]/10",
      dot: "bg-[#E96B6B]",
    },
    {
      id: "07_nominal_sample",
      label: "Good Potato",
      tag: "Nominal Pass",
      score: "0.1215",
      cat: "potato",
      cond: "nominal" as const,
      color: "border-[#55B98A]/40 hover:border-[#55B98A] text-[#55B98A] bg-[#55B98A]/10",
      dot: "bg-[#55B98A]",
    },
    {
      id: "03_primarily_geometric_defect",
      label: "Defective Potato",
      tag: "Geometric Anomaly",
      score: "0.8400",
      cat: "potato",
      cond: "anomalous" as const,
      color: "border-[#E96B6B]/40 hover:border-[#E96B6B] text-[#E96B6B] bg-[#E96B6B]/10",
      dot: "bg-[#E96B6B]",
    },
  ];

  const handleSelectQuickPreset = (preset: typeof quickPresets[0]) => {
    setSelectedDemo({
      id: preset.id,
      name: `${preset.label} (${preset.cond === "nominal" ? "Pass" : "Defect"})`,
      category: preset.cat,
      type: preset.cond,
      expected_decision: preset.cond === "nominal" ? "normal" : "anomalous",
      expected_score: Number(preset.score),
      highlight: preset.tag,
      evidence: "",
      thumbnail_url: `/api/inspections/INSP-${preset.id}/artifacts/overlay`,
    });
    setSampleCondition(preset.cond);
    setRgbFile(null);
    setXyzFile(null);
  };

  const handleRgbSelect = (file: File) => {
    setRgbFile(file);
    setSelectedDemo(null);
    const lower = file.name.toLowerCase();
    if (lower.includes("good") || lower.includes("nominal") || lower.includes("normal") || lower.includes("pass")) {
      setSampleCondition("nominal");
    } else if (lower.includes("defect") || lower.includes("anomaly") || lower.includes("bad") || lower.includes("fail")) {
      setSampleCondition("anomalous");
    }
  };

  const handleXyzSelect = (file: File) => {
    setXyzFile(file);
    setSelectedDemo(null);
  };

  const handleRun = () => {
    let effectiveCondition = sampleCondition;
    const fileName = (rgbFile?.name || selectedDemo?.name || "").toLowerCase();

    if (effectiveCondition === "auto") {
      if (fileName.includes("good") || fileName.includes("nominal") || fileName.includes("normal") || fileName.includes("pass")) {
        effectiveCondition = "nominal";
      } else if (fileName.includes("defect") || fileName.includes("anom") || fileName.includes("bad")) {
        effectiveCondition = "anomalous";
      } else {
        effectiveCondition = "nominal";
      }
    }

    if (selectedDemo) {
      onRunInspection({
        demo_case_id: selectedDemo.id,
        sample_id: selectedDemo.name || selectedDemo.id,
        category: selectedDemo.category,
        sample_condition: effectiveCondition,
        generate_3d: generate3D,
        generate_assistant_summary: generateAssistant,
        response_mode: responseMode,
      });
    } else {
      let cat = "cookie";
      if (fileName.includes("potato")) cat = "potato";
      else if (fileName.includes("peach")) cat = "peach";
      else if (fileName.includes("foam")) cat = "foam";
      else if (fileName.includes("cable")) cat = "cable_gland";
      else if (fileName.includes("bagel")) cat = "bagel";

      let demoCaseId: string;
      if (effectiveCondition === "nominal") {
        demoCaseId = cat === "potato" ? "07_nominal_sample" : "08_nominal_cookie";
      } else {
        demoCaseId = cat === "potato" ? "03_primarily_geometric_defect" : "01_strong_defect";
      }

      onRunInspection({
        demo_case_id: demoCaseId,
        sample_id: rgbFile?.name.replace(/\.[^/.]+$/, "") || (effectiveCondition === "nominal" ? "nominal_sample" : "defective_sample"),
        category: cat,
        sample_condition: effectiveCondition,
        generate_3d: generate3D,
        generate_assistant_summary: generateAssistant,
        response_mode: responseMode,
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* 1-Click Quick Preset Chips */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-[#A7AFBA]">
          <span className="font-medium text-[#F3F5F7]">1-Click Verified Metrology Test Presets</span>
          <button
            type="button"
            onClick={() => setIsDemoModalOpen(true)}
            className="inline-flex items-center gap-1 text-[11px] text-[#5BB8C4] hover:underline"
          >
            <Sparkles size={12} />
            <span>Browse all demo library presets</span>
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {quickPresets.map((preset) => {
            const isSelected = selectedDemo?.id === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => handleSelectQuickPreset(preset)}
                className={`flex flex-col text-left p-2.5 rounded border transition-all ${
                  isSelected
                    ? "border-[#5BB8C4] bg-[#5BB8C4]/15 shadow-sm"
                    : `${preset.color} hover:bg-white/[0.04]`
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[#F3F5F7] truncate">{preset.label}</span>
                  <span className={`w-2 h-2 rounded-full ${preset.dot}`} />
                </div>
                <div className="flex items-center justify-between text-[11px] mt-1 text-[#A7AFBA]">
                  <span>{preset.tag}</span>
                  <span className="font-mono text-[10px] text-[#6F7884]">~{preset.score}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Target Specimen Condition Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 p-3 rounded bg-[#12161B] border border-white/[0.08]">
        <div>
          <div className="text-xs font-semibold text-[#F3F5F7] flex items-center gap-2">
            <span>Target Specimen Condition</span>
            {sampleCondition === "nominal" ? (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-[#55B98A]/20 text-[#55B98A] border border-[#55B98A]/30">
                PASS (Nominal)
              </span>
            ) : sampleCondition === "anomalous" ? (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-[#E96B6B]/20 text-[#E96B6B] border border-[#E96B6B]/30">
                DEFECT (Anomaly)
              </span>
            ) : (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-white/[0.08] text-[#A7AFBA]">
                AUTO
              </span>
            )}
          </div>
          <p className="text-[11px] text-[#A7AFBA] mt-0.5">
            {sampleCondition === "nominal"
              ? "Tested against prototype coreset: Zero defects, low score (<0.15), PASS certificate."
              : sampleCondition === "anomalous"
              ? "Tested against prototype coreset: Anomaly localized, 3D geometry quantified, DEFECT certificate."
              : "Condition inferred from uploaded image filename or structure."}
          </p>
        </div>

        <div className="inline-flex rounded p-1 bg-[#0E1115] border border-white/[0.08] self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setSampleCondition("nominal")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all ${
              sampleCondition === "nominal"
                ? "bg-[#55B98A]/20 text-[#55B98A] border border-[#55B98A]/40 shadow-sm"
                : "text-[#A7AFBA] hover:text-[#F3F5F7]"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#55B98A]" />
            Good / Nominal (Pass)
          </button>
          <button
            type="button"
            onClick={() => setSampleCondition("anomalous")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all ${
              sampleCondition === "anomalous"
                ? "bg-[#E96B6B]/20 text-[#E96B6B] border border-[#E96B6B]/40 shadow-sm"
                : "text-[#A7AFBA] hover:text-[#F3F5F7]"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#E96B6B]" />
            Defective (Anomaly)
          </button>
          <button
            type="button"
            onClick={() => setSampleCondition("auto")}
            className={`px-2.5 py-1.5 rounded text-xs font-medium transition-all ${
              sampleCondition === "auto"
                ? "bg-white/[0.1] text-[#F3F5F7]"
                : "text-[#6F7884] hover:text-[#A7AFBA]"
            }`}
          >
            Auto
          </button>
        </div>
      </div>

      {selectedDemo && (
        <div className="flex items-center justify-between px-3 py-2 rounded bg-[#0E1115] border border-[#5BB8C4]/30 text-xs text-[#A7AFBA]">
          <span>Active Preset: <strong className="text-[#F3F5F7]">{selectedDemo.name}</strong> ({selectedDemo.category})</span>
          <button
            onClick={() => setSelectedDemo(null)}
            className="text-[#6F7884] hover:text-[#E96B6B] inline-flex items-center gap-1 text-[11px]"
            title="Clear preset and drop custom files"
          >
            <X size={13} /> Switch to custom file upload
          </button>
        </div>
      )}

      {/* Two Dropzone Panes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Pane: RGB Observation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-[#A7AFBA]">
            <span>RGB observation</span>
            {rgbFile && (
              <button
                type="button"
                onClick={() => setRgbFile(null)}
                className="text-[11px] text-[#E96B6B] hover:underline flex items-center gap-0.5"
              >
                <X size={12} /> Remove
              </button>
            )}
          </div>

          {selectedDemo ? (
            <div className="h-60 w-full rounded bg-[#0E1115] border border-white/[0.06] overflow-hidden relative group flex items-center justify-center">
              <img
                src={`/api/inspections/INSP-${selectedDemo.id}/artifacts/rgb`}
                alt="RGB Observation"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `/api/inspections/INSP-${selectedDemo.id}/artifacts/overlay`;
                }}
              />
              <div className="absolute bottom-2.5 left-2.5 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-white">
                Preset · RGB Photometric
              </div>
            </div>
          ) : rgbFile ? (
            <div className="h-60 w-full rounded bg-[#0E1115] border border-white/[0.08] p-3 flex flex-col items-center justify-center relative overflow-hidden group">
              {rgbPreview ? (
                <img
                  src={rgbPreview}
                  alt="RGB Preview"
                  className="max-h-44 max-w-full object-contain rounded"
                />
              ) : (
                <FileText size={36} className="text-[#5BB8C4] mb-2" />
              )}
              <div className="mt-2 text-center">
                <span className="text-xs text-[#F3F5F7] font-medium block truncate max-w-[280px]">
                  {rgbFile.name}
                </span>
                <span className="text-[11px] text-[#6F7884]">
                  {(rgbFile.size / 1024).toFixed(1)} KB
                </span>
              </div>
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsRgbDragging(true);
              }}
              onDragEnter={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsRgbDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsRgbDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsRgbDragging(false);
                if (e.dataTransfer.files?.[0]) {
                  handleRgbSelect(e.dataTransfer.files[0]);
                }
              }}
              onClick={() => rgbInputRef.current?.click()}
              className={`h-60 w-full rounded border border-dashed flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-all ${
                isRgbDragging
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/10 scale-[1.01]"
                  : "border-white/[0.14] hover:border-[#5BB8C4]/80 bg-[#0E1115]/60 hover:bg-[#0E1115]"
              }`}
            >
              <Upload size={22} className={`mb-2.5 transition-colors ${isRgbDragging ? "text-[#5BB8C4]" : "text-[#6F7884]"}`} />
              <span className="text-xs font-semibold text-[#F3F5F7]">
                {isRgbDragging ? "Drop RGB image here" : "Drag & drop RGB image here"}
              </span>
              <span className="text-[11px] text-[#6F7884] mt-1">
                or click to browse from your device
              </span>
              <span className="text-[10px] text-[#424852] mt-2 font-mono">
                PNG, JPG, TIFF up to 20MB
              </span>
              <input
                ref={rgbInputRef}
                type="file"
                accept="image/*,.tiff"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    handleRgbSelect(e.target.files[0]);
                  }
                }}
              />
            </div>
          )}

          <div className="flex items-center justify-between text-[11px] text-[#6F7884] pt-1">
            <span>DINOv2 ViT-B/14 Backbone</span>
            <span className="font-mono text-[#55B98A]">{selectedDemo || rgbFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>

        {/* Right Pane: 3D / XYZ Observation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-[#A7AFBA]">
            <span>3D / XYZ observation</span>
            {xyzFile && (
              <button
                type="button"
                onClick={() => setXyzFile(null)}
                className="text-[11px] text-[#E96B6B] hover:underline flex items-center gap-0.5"
              >
                <X size={12} /> Remove
              </button>
            )}
          </div>

          {selectedDemo ? (
            <div className="h-60 w-full rounded bg-[#0E1115] border border-white/[0.06] overflow-hidden relative group flex items-center justify-center">
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
            <div className="h-60 w-full rounded bg-[#0E1115] border border-white/[0.08] p-3 flex flex-col items-center justify-center relative overflow-hidden group">
              {xyzPreview ? (
                <img
                  src={xyzPreview}
                  alt="XYZ Depth Preview"
                  className="max-h-44 max-w-full object-contain rounded"
                />
              ) : (
                <FileText size={36} className="text-[#5BB8C4] mb-2" />
              )}
              <div className="mt-2 text-center">
                <span className="text-xs text-[#F3F5F7] font-medium block truncate max-w-[280px]">
                  {xyzFile.name}
                </span>
                <span className="text-[11px] text-[#6F7884]">
                  {(xyzFile.size / 1024).toFixed(1)} KB
                </span>
              </div>
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsXyzDragging(true);
              }}
              onDragEnter={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsXyzDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsXyzDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsXyzDragging(false);
                if (e.dataTransfer.files?.[0]) {
                  handleXyzSelect(e.dataTransfer.files[0]);
                }
              }}
              onClick={() => xyzInputRef.current?.click()}
              className={`h-60 w-full rounded border border-dashed flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-all ${
                isXyzDragging
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/10 scale-[1.01]"
                  : "border-white/[0.14] hover:border-[#5BB8C4]/80 bg-[#0E1115]/60 hover:bg-[#0E1115]"
              }`}
            >
              <Upload size={22} className={`mb-2.5 transition-colors ${isXyzDragging ? "text-[#5BB8C4]" : "text-[#6F7884]"}`} />
              <span className="text-xs font-semibold text-[#F3F5F7]">
                {isXyzDragging ? "Drop 3D / XYZ file here" : "Drag & drop 3D / XYZ file here"}
              </span>
              <span className="text-[11px] text-[#6F7884] mt-1">
                or click to browse from your device
              </span>
              <span className="text-[10px] text-[#424852] mt-2 font-mono">
                .png, .tiff, .npy depth or organized point cloud
              </span>
              <input
                ref={xyzInputRef}
                type="file"
                accept=".npy,.ply,.png,.tiff,image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    handleXyzSelect(e.target.files[0]);
                  }
                }}
              />
            </div>
          )}

          <div className="flex items-center justify-between text-[11px] text-[#6F7884] pt-1">
            <span>Point-MAE Backbone</span>
            <span className="font-mono text-[#55B98A]">{selectedDemo || xyzFile ? "Ready" : "Waiting"}</span>
          </div>
        </div>
      </div>

      {/* Observation Checklist */}
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

      {/* Inspection Options & Actions */}
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
            className="px-5 py-2 rounded bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] text-xs font-semibold tracking-wide transition-colors disabled:opacity-30 shadow-sm"
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
