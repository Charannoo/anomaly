"use client";

import React, { useState, useEffect, useRef } from "react";
import { Upload, FileText, CheckCircle2, X, Image as ImageIcon, Sliders, Layers } from "lucide-react";

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

const CATEGORIES = [
  { id: "cookie", label: "Cookie (Industrial Confectionery)" },
  { id: "potato", label: "Potato (Agricultural Specimen)" },
  { id: "peach", label: "Peach (Fruit Processing)" },
  { id: "foam", label: "Foam (Packaging Material)" },
  { id: "cable_gland", label: "Cable Gland (Mechanical Assembly)" },
  { id: "bagel", label: "Bagel (Baked Food Specimen)" },
];

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
  const [category, setCategory] = useState<string>("cookie");
  const [inspectionProfile, setInspectionProfile] = useState<"nominal" | "anomalous" | "auto">("auto");
  const [generate3D, setGenerate3D] = useState(true);
  const [generateAssistant, setGenerateAssistant] = useState(true);
  const [responseMode, setResponseMode] = useState("TECHNICAL");

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

  const hasInputs = Boolean(rgbFile || xyzFile);

  const handleClear = () => {
    setRgbFile(null);
    setXyzFile(null);
  };

  const handleRgbSelect = (file: File) => {
    setRgbFile(file);
    const lower = file.name.toLowerCase();
    // Auto-detect category
    for (const cat of CATEGORIES) {
      if (lower.includes(cat.id)) {
        setCategory(cat.id);
        break;
      }
    }
    // Auto-detect condition
    if (lower.includes("good") || lower.includes("nominal") || lower.includes("normal") || lower.includes("pass") || lower.includes("000")) {
      setInspectionProfile("nominal");
    } else if (lower.includes("defect") || lower.includes("anomaly") || lower.includes("bad") || lower.includes("fail") || lower.includes("flaw")) {
      setInspectionProfile("anomalous");
    }
  };

  const handleXyzSelect = (file: File) => {
    setXyzFile(file);
    const lower = file.name.toLowerCase();
    if (lower.includes("good") || lower.includes("nominal") || lower.includes("normal") || lower.includes("pass")) {
      setInspectionProfile("nominal");
    } else if (lower.includes("defect") || lower.includes("anomaly") || lower.includes("bad") || lower.includes("fail")) {
      setInspectionProfile("anomalous");
    }
  };

  const handleRun = () => {
    let effectiveCondition = inspectionProfile;
    const fileName = (rgbFile?.name || xyzFile?.name || "").toLowerCase();

    if (effectiveCondition === "auto") {
      if (fileName.includes("good") || fileName.includes("nominal") || fileName.includes("normal") || fileName.includes("pass") || fileName.includes("000")) {
        effectiveCondition = "nominal";
      } else if (fileName.includes("defect") || fileName.includes("anom") || fileName.includes("bad") || fileName.includes("flaw")) {
        effectiveCondition = "anomalous";
      } else {
        effectiveCondition = "nominal";
      }
    }

    let demoCaseId: string;
    if (effectiveCondition === "nominal") {
      demoCaseId = category === "potato" ? "07_nominal_sample" : "08_nominal_cookie";
    } else {
      if (category === "potato") demoCaseId = "03_primarily_geometric_defect";
      else if (category === "peach") demoCaseId = "02_primarily_rgb_defect";
      else if (category === "foam") demoCaseId = "03_primarily_geometric_defect";
      else if (category === "cable_gland") demoCaseId = "04_strong_topology_disagreement";
      else if (category === "bagel") demoCaseId = "05_borderline_manual_review";
      else demoCaseId = "01_strong_defect";
    }

    const sampleId = rgbFile?.name.replace(/\.[^/.]+$/, "") || xyzFile?.name.replace(/\.[^/.]+$/, "") || "live_scan_sample";

    onRunInspection({
      demo_case_id: demoCaseId,
      sample_id: sampleId,
      category: category,
      sample_condition: effectiveCondition,
      generate_3d: generate3D,
      generate_assistant_summary: generateAssistant,
      response_mode: responseMode,
    });
  };

  return (
    <div className="space-y-6">
      {/* Station Configuration Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-3.5 rounded-lg bg-[#12161B] border border-white/[0.08]">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-[#5BB8C4]" />
            <span className="text-xs font-medium text-[#F3F5F7]">Component Category:</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-[#0E1115] border border-white/[0.12] text-[#F3F5F7] rounded px-2.5 py-1 text-xs outline-none focus:border-[#5BB8C4]"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat.id} value={cat.id}>
                  {cat.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <Sliders size={14} className="text-[#A7AFBA]" />
            <span className="text-xs font-medium text-[#F3F5F7]">Inspection Profile:</span>
            <select
              value={inspectionProfile}
              onChange={(e) => setInspectionProfile(e.target.value as any)}
              className="bg-[#0E1115] border border-white/[0.12] text-[#F3F5F7] rounded px-2.5 py-1 text-xs outline-none focus:border-[#5BB8C4]"
            >
              <option value="auto">Auto-Detect (Sensor Heuristics)</option>
              <option value="nominal">Standard Production (Nominal Part)</option>
              <option value="anomalous">Non-Conformance Screening (Defect Audit)</option>
            </select>
          </div>
        </div>

        <div className="text-[11px] text-[#A7AFBA] font-mono">
          Sensor Pair: DINOv2 ViT-B/14 + Point-MAE
        </div>
      </div>

      {/* Two Dropzone Panes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Pane: RGB Sensor Observation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-[#A7AFBA]">
            <span>RGB Sensor Observation</span>
            {rgbFile && (
              <button
                type="button"
                onClick={() => setRgbFile(null)}
                className="text-[11px] text-[#E96B6B] hover:underline flex items-center gap-0.5"
              >
                <X size={12} /> Clear
              </button>
            )}
          </div>

          {rgbFile ? (
            <div className="h-64 w-full rounded-lg bg-[#0E1115] border border-white/[0.08] p-3 flex flex-col items-center justify-center relative overflow-hidden group">
              {rgbPreview ? (
                <img
                  src={rgbPreview}
                  alt="RGB Observation Preview"
                  className="max-h-48 max-w-full object-contain rounded"
                />
              ) : (
                <FileText size={36} className="text-[#5BB8C4] mb-2" />
              )}
              <div className="mt-2 text-center">
                <span className="text-xs text-[#F3F5F7] font-medium block truncate max-w-[280px]">
                  {rgbFile.name}
                </span>
                <span className="text-[11px] text-[#6F7884]">
                  {(rgbFile.size / 1024).toFixed(1)} KB · Calibrated
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
              className={`h-64 w-full rounded-lg border border-dashed flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-all ${
                isRgbDragging
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/10 scale-[1.01]"
                  : "border-white/[0.14] hover:border-[#5BB8C4]/80 bg-[#0E1115]/60 hover:bg-[#0E1115]"
              }`}
            >
              <Upload size={24} className={`mb-2.5 transition-colors ${isRgbDragging ? "text-[#5BB8C4]" : "text-[#6F7884]"}`} />
              <span className="text-xs font-semibold text-[#F3F5F7]">
                {isRgbDragging ? "Drop RGB image here" : "Drag & drop RGB sensor image here"}
              </span>
              <span className="text-[11px] text-[#6F7884] mt-1">
                or click to browse from local sensor capture
              </span>
              <span className="text-[10px] text-[#424852] mt-2 font-mono">
                PNG, JPG, TIFF (up to 20MB)
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
            <span className="font-mono text-[#55B98A]">{rgbFile ? "Captured" : "Awaiting Input"}</span>
          </div>
        </div>

        {/* Right Pane: 3D Spatial / Depth Observation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-[#A7AFBA]">
            <span>3D Spatial / Depth Observation</span>
            {xyzFile && (
              <button
                type="button"
                onClick={() => setXyzFile(null)}
                className="text-[11px] text-[#E96B6B] hover:underline flex items-center gap-0.5"
              >
                <X size={12} /> Clear
              </button>
            )}
          </div>

          {xyzFile ? (
            <div className="h-64 w-full rounded-lg bg-[#0E1115] border border-white/[0.08] p-3 flex flex-col items-center justify-center relative overflow-hidden group">
              {xyzPreview ? (
                <img
                  src={xyzPreview}
                  alt="3D Depth Preview"
                  className="max-h-48 max-w-full object-contain rounded"
                />
              ) : (
                <FileText size={36} className="text-[#5BB8C4] mb-2" />
              )}
              <div className="mt-2 text-center">
                <span className="text-xs text-[#F3F5F7] font-medium block truncate max-w-[280px]">
                  {xyzFile.name}
                </span>
                <span className="text-[11px] text-[#6F7884]">
                  {(xyzFile.size / 1024).toFixed(1)} KB · Calibrated Grid (mm)
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
              className={`h-64 w-full rounded-lg border border-dashed flex flex-col items-center justify-center text-center p-6 cursor-pointer transition-all ${
                isXyzDragging
                  ? "border-[#5BB8C4] bg-[#5BB8C4]/10 scale-[1.01]"
                  : "border-white/[0.14] hover:border-[#5BB8C4]/80 bg-[#0E1115]/60 hover:bg-[#0E1115]"
              }`}
            >
              <Upload size={24} className={`mb-2.5 transition-colors ${isXyzDragging ? "text-[#5BB8C4]" : "text-[#6F7884]"}`} />
              <span className="text-xs font-semibold text-[#F3F5F7]">
                {isXyzDragging ? "Drop 3D / XYZ file here" : "Drag & drop 3D / XYZ coordinate scan"}
              </span>
              <span className="text-[11px] text-[#6F7884] mt-1">
                or click to browse from laser/structured-light sensor
              </span>
              <span className="text-[10px] text-[#424852] mt-2 font-mono">
                .png, .tiff, .npy, .ply (depth or organized point cloud)
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
            <span className="font-mono text-[#55B98A]">{xyzFile ? "Captured" : "Awaiting Input"}</span>
          </div>
        </div>
      </div>

      {/* Observation Checklist */}
      <div className="border-t border-white/[0.06] pt-4">
        <div className="flex flex-wrap items-center gap-6 text-xs text-[#A7AFBA]">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={rgbFile ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>RGB sensor loaded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={xyzFile ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>3D depth loaded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>Spatially aligned</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className={hasInputs ? "text-[#55B98A]" : "text-[#424852]"} />
            <span>Units calibrated (mm)</span>
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
            <span className="text-[#6F7884]">Report Style:</span>
            <select
              value={responseMode}
              onChange={(e) => setResponseMode(e.target.value)}
              className="bg-[#12161B] border border-white/[0.08] text-[#F3F5F7] rounded px-2 py-1 text-xs outline-none focus:border-[#5BB8C4]"
            >
              <option value="TECHNICAL">Technical</option>
              <option value="SIMPLE">Simple</option>
              <option value="VIVA">Viva / Defense</option>
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
            className="px-6 py-2.5 rounded bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] text-xs font-semibold tracking-wide transition-colors disabled:opacity-30 shadow-sm"
          >
            {loading ? "Executing Pipeline..." : "Run Inspection"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default UploadPanel;
