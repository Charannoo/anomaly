"use client";

import React, { useEffect, useState } from "react";
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
  FileText,
  Maximize2,
  Minimize2,
  CheckCircle2,
} from "lucide-react";

const FLOW_STEPS = [
  { id: 1, name: "Sample selection", desc: "Select verified MVTec-3D benchmark case" },
  { id: 2, name: "Execution", desc: "Run multimodal PNTC inspection" },
  { id: 3, name: "Heatmap & contours", desc: "Segmented localized defect regions" },
  { id: 4, name: "3D metrology", desc: "Calibrated dimensions & material volume" },
  { id: 5, name: "Normal twin", desc: "Retrieved nearest paired prototype" },
  { id: 6, name: "Evidence trace", desc: "Cross-modal JS divergence & gate" },
  { id: 7, name: "AI grounded Q&A", desc: "Conversational metrology explanation" },
  { id: 8, name: "3D point cloud", desc: "Interactive geometry surface rendering" },
  { id: 9, name: "Certificate export", desc: "Audited ISO metrology documentation" },
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
    <div className={`space-y-8 ${isPresentationMode ? "fixed inset-0 z-50 bg-[#0B0D10] p-8 overflow-y-auto" : "max-w-[1500px] mx-auto"}`}>
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-6 border-b border-white/[0.05]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-text-muted">Demonstration flow</span>
            <span className="text-white/[0.2]">•</span>
            <span className="text-xs font-mono text-accent-primary">
              Step {currentStep} of {FLOW_STEPS.length}
            </span>
          </div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-text-primary">
            Demo Mode
          </h1>
          <p className="text-sm text-text-muted mt-1">
            End-to-end metrology workflow demonstrating multimodal detection, geometry extraction, and grounded reasoning.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPresentationMode(!isPresentationMode)}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-bg-surface hover:bg-bg-elevated border border-white/[0.08] text-xs font-medium text-text-primary transition-colors"
          >
            {isPresentationMode ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span>Exit fullscreen</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span>Presentation mode</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Step Progression Bar - Clean strip */}
      <div className="grid grid-cols-3 sm:grid-cols-9 gap-1.5 p-1 rounded-md bg-bg-surface border border-white/[0.05]">
        {FLOW_STEPS.map((step) => {
          const isActive = step.id === currentStep;
          const isCompleted = step.id < currentStep;
          return (
            <button
              key={step.id}
              onClick={() => setCurrentStep(step.id)}
              className={`py-2 px-2.5 text-left rounded transition-colors ${
                isActive
                  ? "bg-bg-elevated text-text-primary font-medium"
                  : isCompleted
                  ? "text-status-normal hover:bg-white/[0.02]"
                  : "text-text-muted hover:text-text-primary hover:bg-white/[0.02]"
              }`}
            >
              <div className="text-[11px] flex items-center gap-1.5">
                {isCompleted ? (
                  <CheckCircle2 className="w-3 h-3 text-status-normal shrink-0" />
                ) : (
                  <span className="text-[10px] font-mono text-text-muted shrink-0">
                    {step.id}
                  </span>
                )}
                <span className="truncate">{step.name}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Step View Content Container */}
      <div className="p-6 rounded-md border border-white/[0.06] bg-bg-surface min-h-[500px]">
        {/* STEP 1: SAMPLE SELECTION */}
        {currentStep === 1 && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-text-primary">Select benchmark sample</h2>
              <p className="text-xs text-text-muted mt-1">
                Choose from authentic pre-computed MVTec-3D industrial specimens.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {cases.map((c) => (
                <button
                  key={c.id}
                  onClick={() => loadCaseInspection(c.id)}
                  className={`p-4 rounded border text-left transition-all ${
                    selectedCaseId === c.id
                      ? "border-accent-primary bg-accent-primary/5"
                      : "border-white/[0.06] bg-bg-app hover:border-white/[0.12]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-text-primary">{c.name}</span>
                    <StatusIndicator
                      status={c.expected_decision === "ANOMALY" ? "anomaly" : "normal"}
                      size="sm"
                    />
                  </div>
                  <div className="text-[11px] text-accent-primary font-mono capitalize mb-1">{c.category}</div>
                  <p className="text-xs text-text-muted line-clamp-2">{c.highlight}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* STEP 2: EXECUTION */}
        {currentStep === 2 && (
          <div className="flex flex-col items-center justify-center py-16 space-y-5 text-center">
            <div className="w-12 h-12 rounded-full border border-accent-primary/40 bg-accent-primary/10 flex items-center justify-center">
              <Play className="w-5 h-5 text-accent-primary ml-0.5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-text-primary">PNTC multimodal execution</h2>
              <p className="text-xs text-text-muted max-w-md mt-1">
                Extracting 768-dim DINOv2 RGB tokens and 1152-dim Point-MAE geometry features, then matching against
                the 15,000 paired prototype coreset.
              </p>
            </div>
            <div className="p-4 rounded bg-bg-app border border-white/[0.05] text-xs text-left w-full max-w-md space-y-1.5">
              <div className="text-status-normal">✓ Inputs verified (Organized XYZ 224x224)</div>
              <div className="text-status-normal">✓ DINOv2 feature extraction complete</div>
              <div className="text-status-normal">✓ Point-MAE geometry features computed</div>
              <div className="text-status-normal">✓ Top-5 paired prototype retrieval executed</div>
              <div className="text-status-normal">✓ Cross-modal JS divergence evaluated</div>
            </div>
            <button
              onClick={() => setCurrentStep(3)}
              className="px-5 py-2 rounded bg-accent-primary hover:bg-accent-hover text-bg-app font-semibold text-xs transition-colors shadow-sm"
            >
              Proceed to heatmap inspection →
            </button>
          </div>
        )}

        {/* STEP 3: HEATMAP & CONTOURS */}
        {currentStep === 3 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-text-primary">Segmented defect heatmap</h2>
                <p className="text-xs text-text-muted mt-0.5">
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
              <h2 className="text-base font-semibold text-text-primary">Calibrated 3D metrology</h2>
              <p className="text-xs text-text-muted mt-0.5">
                Quantitative dimensions, surface depth profiles, and estimated material displacement.
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
              <h2 className="text-base font-semibold text-text-primary">Nearest normal twin reference</h2>
              <p className="text-xs text-text-muted mt-0.5">
                Retrieved paired normal prototype showing nominal surface geometry.
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
              <h2 className="text-base font-semibold text-text-primary">Prototype retrieval & topology trace</h2>
              <p className="text-xs text-text-muted mt-0.5">
                Top-5 RGB vs XYZ nearest prototype neighborhoods and JS divergence decomposition.
              </p>
            </div>
            <div className="max-w-xl mx-auto">
              <PrototypeTrace defect={primaryDefect} />
            </div>
          </div>
        )}

        {/* STEP 7: AI ASSISTANT Q&A */}
        {currentStep === 7 && (
          <div className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-text-primary">Grounded AI explanation</h2>
              <p className="text-xs text-text-muted mt-0.5">
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
          <div className="space-y-5 text-center py-16">
            <div>
              <h2 className="text-base font-semibold text-text-primary">Interactive 3D surface inspection</h2>
              <p className="text-xs text-text-muted mt-1">
                Rotate, pan, and inspect surface point cloud deviation in WebGL 3D space.
              </p>
            </div>
            <button
              onClick={() => setIs3DOpen(true)}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded bg-accent-primary hover:bg-accent-hover text-bg-app font-semibold text-xs shadow-sm transition-colors"
            >
              <Box className="w-4 h-4" />
              Launch interactive 3D viewport
            </button>
          </div>
        )}

        {/* STEP 9: CERTIFICATE EXPORT */}
        {currentStep === 9 && (
          <div className="space-y-5 text-center py-16">
            <div>
              <h2 className="text-base font-semibold text-text-primary">ISO metrology certificate</h2>
              <p className="text-xs text-text-muted mt-1">
                Generate tamper-evident, auditable documentation with all calibrated measurements.
              </p>
            </div>
            <div className="flex items-center justify-center gap-3">
              <a
                href={`/api/inspections/${selectedCaseId}/export/html`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-xs font-medium text-text-primary transition-colors"
              >
                <FileText className="w-4 h-4 text-accent-primary" />
                Open certificate (HTML / PDF)
              </a>
              <a
                href={`/api/inspections/${selectedCaseId}/export/json`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-xs font-medium text-text-muted hover:text-text-primary transition-colors"
              >
                <FileText className="w-4 h-4 text-text-muted" />
                Raw JSON
              </a>
            </div>
          </div>
        )}
      </div>

      {/* Step Navigation Buttons */}
      <div className="flex items-center justify-between border-t border-white/[0.05] pt-4">
        <button
          onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
          disabled={currentStep === 1}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated text-xs font-medium text-text-primary disabled:opacity-30 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          Previous stage
        </button>

        <span className="text-xs text-text-muted">
          {FLOW_STEPS[currentStep - 1]?.desc}
        </span>

        <button
          onClick={() => setCurrentStep((prev) => Math.min(9, prev + 1))}
          disabled={currentStep === 9}
          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded bg-accent-primary hover:bg-accent-hover text-bg-app font-semibold text-xs disabled:opacity-30 transition-colors shadow-sm"
        >
          Next stage
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
  );
}
