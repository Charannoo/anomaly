"use client";

import React, { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchDemoCases, fetchInspection, DemoCase } from "@/lib/api";
import StatusIndicator from "@/components/StatusIndicator";
import ImageViewer from "@/components/ImageViewer";
import DefectInspector from "@/components/DefectInspector";
import NormalTwinViewer from "@/components/NormalTwinViewer";
import PrototypeTrace from "@/components/PrototypeTrace";
import AssistantPanel from "@/components/AssistantPanel";
import ThreeDModal from "@/components/ThreeDModal";
import {
  Play,
  ChevronRight,
  ChevronLeft,
  Box,
  Layers,
  FileText,
  MessageSquare,
  Sparkles,
  Maximize2,
  Minimize2,
  CheckCircle2,
} from "lucide-react";

const FLOW_STEPS = [
  { id: 1, name: "Sample Selection", desc: "Select verified MVTec-3D benchmark case" },
  { id: 2, name: "Execution", desc: "Run multimodal PNTC inspection" },
  { id: 3, name: "Heatmap & Contours", desc: "Segmented localized defect regions" },
  { id: 4, name: "3D Metrology", desc: "Calibrated dimensions & material volume" },
  { id: 5, name: "Normal Twin", desc: "Retrieved nearest paired prototype" },
  { id: 6, name: "Evidence Trace", desc: "Cross-modal JS divergence & gate" },
  { id: 7, name: "AI Grounded Q&A", desc: "Conversational metrology explanation" },
  { id: 8, name: "3D Point Cloud", desc: "Interactive geometry surface rendering" },
  { id: 9, name: "Certificate Export", desc: "Audited ISO metrology documentation" },
];

export default function DemoPage() {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("01_strong_defect");
  const [currentStep, setCurrentStep] = useState(1);
  const [inspection, setInspection] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [is3DOpen, setIs3DOpen] = useState(false);
  const [isPresentationMode, setIsPresentationMode] = useState(false);

  useEffect(() => {
    fetchDemoCases()
      .then((items) => {
        setCases(items);
        if (items.length > 0) {
          loadCaseInspection(items[0].id);
        }
      })
      .catch((err) => console.error("Failed to load demo cases", err));
  }, []);

  const loadCaseInspection = async (id: string) => {
    setLoading(true);
    try {
      const data = await fetchInspection(id);
      setInspection(data);
      setSelectedCaseId(id);
    } catch (err) {
      console.error("Failed to load case inspection", err);
    } finally {
      setLoading(false);
    }
  };

  const currentCase = cases.find((c) => c.id === selectedCaseId);
  const report = inspection?.report || {};
  const defects = report.defects || [];
  const primaryDefect = defects[0] || null;

  return (
    <AppShell
      title="Faculty Presentation Mode"
      breadcrumb="End-to-End Multimodal Inspection Demonstration"
    >
      <div className={`space-y-6 ${isPresentationMode ? "fixed inset-0 z-50 bg-[#0B0D10] p-6 overflow-y-auto" : ""}`}>
        {/* Top Control Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/[0.08] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F7]">
                PNTC Faculty Demonstration Flow
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#5BB8C4]/10 text-[#5BB8C4] border border-[#5BB8C4]/30">
                STAGE {currentStep} OF 9
              </span>
            </div>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Structured sequence showcasing real multimodal detection, metrology, normal twin, and grounded reasoning.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPresentationMode(!isPresentationMode)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/[0.08] bg-[#15191F] hover:bg-[#191E25] text-xs font-medium text-[#F3F5F7] transition-colors"
            >
              {isPresentationMode ? (
                <>
                  <Minimize2 className="w-3.5 h-3.5" />
                  Exit Fullscreen
                </>
              ) : (
                <>
                  <Maximize2 className="w-3.5 h-3.5" />
                  Presentation Mode
                </>
              )}
            </button>
          </div>
        </div>

        {/* Step Progression Strip */}
        <div className="grid grid-cols-3 sm:grid-cols-9 gap-1.5 p-1.5 rounded-lg bg-[#101318] border border-white/[0.06]">
          {FLOW_STEPS.map((step) => {
            const isActive = step.id === currentStep;
            const isCompleted = step.id < currentStep;
            return (
              <button
                key={step.id}
                onClick={() => setCurrentStep(step.id)}
                className={`py-2 px-2 text-left rounded transition-all ${
                  isActive
                    ? "bg-[#191E25] border border-[#5BB8C4]/40 text-[#F3F5F7] shadow-sm"
                    : isCompleted
                    ? "text-[#55B98A] hover:bg-white/[0.02]"
                    : "text-[#6F7884] hover:bg-white/[0.02]"
                }`}
              >
                <div className="text-[10px] font-mono font-medium flex items-center gap-1">
                  {isCompleted ? <CheckCircle2 className="w-3 h-3 text-[#55B98A]" /> : <span>#{step.id}</span>}
                  <span className="truncate">{step.name}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Step View Content Container */}
        <div className="p-6 rounded-lg border border-white/[0.08] bg-[#15191F] min-h-[520px]">
          {/* STEP 1: SAMPLE SELECTION */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 1: Select Verified Benchmark Sample</h2>
                <p className="text-xs text-[#A7AFBA] mt-0.5">
                  Choose from authentic pre-computed MVTec-3D industrial specimens.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {cases.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => loadCaseInspection(c.id)}
                    className={`p-4 rounded-md border text-left transition-all ${
                      selectedCaseId === c.id
                        ? "border-[#5BB8C4] bg-[#5BB8C4]/5"
                        : "border-white/[0.08] bg-[#101318] hover:border-white/[0.14]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-[#F3F5F7]">{c.name}</span>
                      <StatusIndicator
                        status={c.expected_decision === "ANOMALY" ? "anomaly" : "normal"}
                        size="sm"
                      />
                    </div>
                    <div className="text-[11px] text-[#5BB8C4] font-mono mb-1">{c.category.toUpperCase()}</div>
                    <p className="text-[11px] text-[#A7AFBA] line-clamp-2">{c.highlight}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STEP 2: EXECUTION */}
          {currentStep === 2 && (
            <div className="flex flex-col items-center justify-center py-12 space-y-4 text-center">
              <div className="w-12 h-12 rounded-full border border-[#5BB8C4]/40 bg-[#5BB8C4]/10 flex items-center justify-center">
                <Play className="w-5 h-5 text-[#5BB8C4]" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-[#F3F5F7]">PNTC Multimodal Execution</h2>
                <p className="text-xs text-[#A7AFBA] max-w-md mt-1">
                  Extracting 768-dim DINOv2 RGB tokens and 1152-dim Point-MAE geometry features, then matching against
                  the 15,000 paired prototype coreset.
                </p>
              </div>
              <div className="p-4 rounded-md bg-[#101318] border border-white/[0.06] font-mono text-xs text-left w-full max-w-md space-y-1">
                <div className="text-[#55B98A]">✓ Inputs verified (Organized XYZ 224x224)</div>
                <div className="text-[#55B98A]">✓ DINOv2 feature extraction complete</div>
                <div className="text-[#55B98A]">✓ Point-MAE geometry features computed</div>
                <div className="text-[#55B98A]">✓ Top-5 paired prototype retrieval executed</div>
                <div className="text-[#55B98A]">✓ Cross-modal JS divergence evaluated</div>
              </div>
              <button
                onClick={() => setCurrentStep(3)}
                className="px-4 py-2 rounded-md bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] font-semibold text-xs transition-colors"
              >
                Proceed to Heatmap Inspection →
              </button>
            </div>
          )}

          {/* STEP 3: HEATMAP & CONTOURS */}
          {currentStep === 3 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 3: Segmented Defect Heatmap</h2>
                  <p className="text-xs text-[#A7AFBA]">
                    Overlaying PNTC anomaly score distribution on photometric RGB observation.
                  </p>
                </div>
                {inspection && (
                  <StatusIndicator
                    status={inspection.decision === "ANOMALY" ? "anomaly" : "normal"}
                    score={inspection.anomaly_score}
                  />
                )}
              </div>
              <ImageViewer inspectionId={selectedCaseId} report={report} />
            </div>
          )}

          {/* STEP 4: 3D METROLOGY */}
          {currentStep === 4 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 4: Calibrated 3D Metrology</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Quantitative dimensions, surface depth profiles, and estimated material volume displacement.
                </p>
              </div>
              <div className="max-w-xl mx-auto">
                <DefectInspector defect={primaryDefect} onSelectDefect={() => {}} />
              </div>
            </div>
          )}

          {/* STEP 5: NORMAL TWIN */}
          {currentStep === 5 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 5: Nearest Normal Twin Reference</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Retrieved paired normal prototype showing how the anomalous region ought to appear.
                </p>
              </div>
              <div className="max-w-xl mx-auto">
                <NormalTwinViewer
                  inspectionId={selectedCaseId}
                  normalTwin={report.normal_twin}
                  category={inspection?.category || "part"}
                />
              </div>
            </div>
          )}

          {/* STEP 6: EVIDENCE TRACE */}
          {currentStep === 6 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 6: Prototype Retrieval & Topology Trace</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Examining top-5 RGB vs XYZ nearest prototype neighborhoods and JS divergence decomposition.
                </p>
              </div>
              <div className="max-w-xl mx-auto">
                <PrototypeTrace trace={report.prototype_trace} />
              </div>
            </div>
          )}

          {/* STEP 7: AI ASSISTANT Q&A */}
          {currentStep === 7 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 7: Grounded AI Explanation</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Conversational assistant synthesizing verified measurements without fabrication.
                </p>
              </div>
              <div className="max-w-xl mx-auto">
                <AssistantPanel sampleId={selectedCaseId} />
              </div>
            </div>
          )}

          {/* STEP 8: 3D POINT CLOUD */}
          {currentStep === 8 && (
            <div className="space-y-4 text-center py-8">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 8: Interactive 3D Defect Inspection</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Rotate, zoom, and inspect surface point cloud deviation in WebGL 3D space.
                </p>
              </div>
              <button
                onClick={() => setIs3DOpen(true)}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] font-semibold text-xs shadow-md transition-colors"
              >
                <Box className="w-4 h-4" />
                Launch Interactive 3D Metrology Viewport
              </button>
            </div>
          )}

          {/* STEP 9: CERTIFICATE EXPORT */}
          {currentStep === 9 && (
            <div className="space-y-4 text-center py-8">
              <div>
                <h2 className="text-sm font-semibold text-[#F3F5F7]">Step 9: ISO Metrology Certificate</h2>
                <p className="text-xs text-[#A7AFBA]">
                  Generate tamper-evident, auditable documentation with all calibrated measurements.
                </p>
              </div>
              <div className="flex items-center justify-center gap-3">
                <a
                  href={`/api/inspections/${selectedCaseId}/export/html`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/[0.08] bg-[#101318] hover:bg-[#191E25] text-xs font-medium text-[#F3F5F7]"
                >
                  <FileText className="w-4 h-4 text-[#5BB8C4]" />
                  Open ISO Certificate (HTML / PDF)
                </a>
                <a
                  href={`/api/inspections/${selectedCaseId}/export/json`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/[0.08] bg-[#101318] hover:bg-[#191E25] text-xs font-medium text-[#F3F5F7]"
                >
                  <FileText className="w-4 h-4 text-[#A7AFBA]" />
                  Download Raw JSON
                </a>
              </div>
            </div>
          )}
        </div>

        {/* Step Navigation Buttons */}
        <div className="flex items-center justify-between border-t border-white/[0.08] pt-4">
          <button
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            disabled={currentStep === 1}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/[0.08] bg-[#15191F] hover:bg-[#191E25] text-xs font-medium text-[#F3F5F7] disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous Stage
          </button>

          <span className="text-xs text-[#6F7884] font-mono">
            {FLOW_STEPS[currentStep - 1]?.desc}
          </span>

          <button
            onClick={() => setCurrentStep((prev) => Math.min(9, prev + 1))}
            disabled={currentStep === 9}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#5BB8C4] hover:bg-[#71C7D1] text-[#0B0D10] font-semibold text-xs disabled:opacity-30 transition-colors"
          >
            Next Stage
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* 3D Modal */}
        <ThreeDModal
          isOpen={is3DOpen}
          onClose={() => setIs3DOpen(false)}
          inspectionId={selectedCaseId}
          defect={primaryDefect}
        />
      </div>
    </AppShell>
  );
}
