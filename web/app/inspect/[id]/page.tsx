"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Download, Cuboid, MessageSquare, ChevronRight, AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import { fetchInspection, InspectionDetail } from "@/lib/api";
import { ImageViewer } from "@/components/ImageViewer";
import { DefectInspector } from "@/components/DefectInspector";
import { AssistantPanel } from "@/components/AssistantPanel";
import { NormalTwinViewer } from "@/components/NormalTwinViewer";
import { PrototypeTrace } from "@/components/PrototypeTrace";
import { QualityPanel } from "@/components/QualityPanel";
import { ThreeDModal } from "@/components/ThreeDModal";
import { cn, formatMetric } from "@/lib/utils";

export default function InspectionResultPage() {
  const params = useParams();
  const inspectionId = params.id as string;

  const [inspection, setInspection] = useState<InspectionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedDefectId, setSelectedDefectId] = useState<number>(1);
  const [activeRightTab, setActiveRightTab] = useState<"explain" | "normal_twin" | "trace" | "quality">("explain");
  const [is3DModalOpen, setIs3DModalOpen] = useState(false);
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchInspection(inspectionId)
      .then((data) => {
        setInspection(data);
        if (data.report?.defects?.length > 0) {
          setSelectedDefectId(data.report.defects[0].id);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [inspectionId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-24 text-text-muted text-xs font-mono">
        Loading verified inspection certificate...
      </div>
    );
  }

  if (!inspection) {
    return (
      <div className="max-w-xl mx-auto p-12 text-center border border-border-default rounded bg-bg-panel space-y-3">
        <h2 className="text-base font-semibold text-text-primary">Inspection Not Found</h2>
        <p className="text-xs text-text-muted">The requested inspection certificate could not be resolved from local persistence.</p>
        <Link href="/inspections" className="inline-block text-xs text-accent-primary hover:underline font-mono">
          ← Back to Inspection History
        </Link>
      </div>
    );
  }

  const report = inspection.report || {};
  const defects = report.defects || [];
  const activeDefect = defects.find((d: any) => d.id === selectedDefectId) || defects[0];
  const pntc = report.pntc || {};
  const decision = pntc.decision || "normal";
  const isAnomaly = decision === "anomalous";
  const isReview = inspection.manual_review_recommended;

  // Handle assistant emitted UI action
  const handleActionTriggered = (actionType: string) => {
    if (actionType === "SHOW_NORMAL_TWIN") {
      setActiveRightTab("normal_twin");
    } else if (actionType === "SHOW_PROTOTYPE_TRACE") {
      setActiveRightTab("trace");
    } else if (actionType === "SHOW_3D_VIEW") {
      setIs3DModalOpen(true);
    }
  };

  return (
    <div className="space-y-4 max-w-[1920px] mx-auto h-[calc(100vh-80px)] flex flex-col">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3 border-b border-border-default flex-shrink-0">
        <div>
          <div className="flex items-center gap-2 text-xs text-text-muted font-mono mb-1">
            <Link href="/inspections" className="hover:text-text-primary">
              Inspections
            </Link>
            <ChevronRight size={11} />
            <span className="text-text-secondary">{inspection.category}</span>
            <ChevronRight size={11} />
            <span className="text-accent-primary">{inspection.id}</span>
          </div>
          <h1 className="text-lg font-semibold tracking-tight text-text-primary flex items-center gap-2">
            <span>Inspection {inspection.id}</span>
            <span className="text-xs font-mono font-normal text-text-muted">
              ({inspection.sample_id})
            </span>
          </h1>
        </div>

        {/* Top Action Toolbar */}
        <div className="flex items-center gap-2 relative">
          <button
            onClick={() => setIs3DModalOpen(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-DEFAULT bg-bg-panel border border-border-default hover:border-accent-primary text-text-primary hover:text-accent-primary text-xs font-medium transition-colors"
          >
            <Cuboid size={14} className="text-accent-primary" />
            <span>3D Metrology View</span>
          </button>

          {/* Export Dropdown Menu */}
          <div className="relative">
            <button
              onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-DEFAULT bg-bg-panel border border-border-default hover:border-border-emphasized text-text-primary text-xs font-medium transition-colors"
            >
              <Download size={14} />
              <span>Export Report</span>
            </button>

            {isExportMenuOpen && (
              <div className="absolute right-0 mt-1 w-48 border border-border-default rounded bg-bg-panel shadow-panel py-1 z-50 text-xs font-mono">
                <a
                  href={`/api/inspections/${inspectionId}/export/html`}
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3 py-1.5 hover:bg-bg-surface text-text-secondary hover:text-text-primary"
                >
                  Printable ISO HTML / PDF
                </a>
                <a
                  href={`/api/inspections/${inspectionId}/export/json`}
                  download
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3 py-1.5 hover:bg-bg-surface text-text-secondary hover:text-text-primary"
                >
                  Machine JSON Output
                </a>
                <a
                  href={`/api/inspections/${inspectionId}/export/text`}
                  download
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3 py-1.5 hover:bg-bg-surface text-text-secondary hover:text-text-primary"
                >
                  Technical Report (TXT)
                </a>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Restrained Status Banner (with accent line) */}
      <div
        className={cn(
          "flex flex-wrap items-center justify-between gap-4 p-3 rounded-md bg-bg-panel border border-border-default flex-shrink-0 text-xs font-mono",
          isReview
            ? "border-l-4 border-l-status-review"
            : isAnomaly
            ? "border-l-4 border-l-status-anomaly"
            : "border-l-4 border-l-status-normal"
        )}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                isReview ? "bg-status-review" : isAnomaly ? "bg-status-anomaly" : "bg-status-normal"
              )}
            />
            <span
              className={cn(
                "font-bold tracking-wide",
                isReview ? "text-status-review" : isAnomaly ? "text-status-anomaly" : "text-status-normal"
              )}
            >
              {report.inspection_status || (isAnomaly ? "DEFECT DETECTED" : "NORMAL")}
            </span>
          </div>
          <span className="text-text-muted">|</span>
          <span className="text-text-secondary">
            PNTC Score:{" "}
            <strong className="text-text-primary tabular-nums font-semibold">
              {formatMetric(pntc.score, 4)}
            </strong>
          </span>
          <span className="text-text-muted">|</span>
          <span className="text-text-secondary">
            Threshold: <strong className="text-text-primary tabular-nums">{pntc.threshold || 0.5}</strong>
          </span>
        </div>

        <div className="flex items-center gap-4 text-text-muted">
          <div>
            Certainty: <strong className="text-text-primary">{report.decision_certainty || "High"}</strong>
          </div>
          <div>
            Detected Regions:{" "}
            <strong className="text-text-primary tabular-nums">{defects.length}</strong>
          </div>
          <div>
            Execution: <strong className="text-text-primary font-mono">{formatMetric(inspection.execution_time_ms, 1)} ms</strong>
          </div>
        </div>
      </div>

      {/* Main Three-Column Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0 overflow-hidden">
        {/* LEFT COLUMN: Visual Inspection (approx 45% -> 5 cols) */}
        <div className="lg:col-span-5 h-full overflow-hidden flex flex-col">
          <ImageViewer
            inspectionId={inspectionId}
            defects={defects}
            selectedDefectId={selectedDefectId}
            onSelectDefect={(id) => setSelectedDefectId(id)}
            onOpen3DView={() => setIs3DModalOpen(true)}
          />
        </div>

        {/* CENTER COLUMN: Defect Inspector (approx 30% -> 4 cols) */}
        <div className="lg:col-span-4 h-full overflow-y-auto">
          <DefectInspector
            defect={activeDefect}
            overallDecision={decision}
            onViewTrace={() => setActiveRightTab("trace")}
          />
        </div>

        {/* RIGHT COLUMN: Inspector Drawer Context (approx 25% -> 3 cols) */}
        <div className="lg:col-span-3 h-full border border-border-default rounded-md bg-bg-panel flex flex-col overflow-hidden">
          {/* Drawer Tabs */}
          <div className="flex items-center border-b border-border-default bg-bg-subtle px-2 py-1.5 gap-1 font-mono text-[11px] overflow-x-auto flex-shrink-0">
            <button
              onClick={() => setActiveRightTab("explain")}
              className={cn(
                "px-2 py-1 rounded transition-colors whitespace-nowrap",
                activeRightTab === "explain"
                  ? "bg-bg-active text-accent-primary font-bold border border-border-subtle"
                  : "text-text-muted hover:text-text-primary hover:bg-bg-surface"
              )}
            >
              Explain
            </button>
            <button
              onClick={() => setActiveRightTab("normal_twin")}
              className={cn(
                "px-2 py-1 rounded transition-colors whitespace-nowrap",
                activeRightTab === "normal_twin"
                  ? "bg-bg-active text-accent-primary font-bold border border-border-subtle"
                  : "text-text-muted hover:text-text-primary hover:bg-bg-surface"
              )}
            >
              Normal Twin
            </button>
            <button
              onClick={() => setActiveRightTab("trace")}
              className={cn(
                "px-2 py-1 rounded transition-colors whitespace-nowrap",
                activeRightTab === "trace"
                  ? "bg-bg-active text-accent-primary font-bold border border-border-subtle"
                  : "text-text-muted hover:text-text-primary hover:bg-bg-surface"
              )}
            >
              Trace
            </button>
            <button
              onClick={() => setActiveRightTab("quality")}
              className={cn(
                "px-2 py-1 rounded transition-colors whitespace-nowrap",
                activeRightTab === "quality"
                  ? "bg-bg-active text-accent-primary font-bold border border-border-subtle"
                  : "text-text-muted hover:text-text-primary hover:bg-bg-surface"
              )}
            >
              Quality
            </button>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 p-3.5 overflow-y-auto">
            {activeRightTab === "explain" && (
              <AssistantPanel
                sampleId={inspectionId}
                defectId={selectedDefectId}
                onActionTriggered={handleActionTriggered}
              />
            )}
            {activeRightTab === "normal_twin" && (
              <NormalTwinViewer
                inspectionId={inspectionId}
                defect={activeDefect}
              />
            )}
            {activeRightTab === "trace" && (
              <PrototypeTrace defect={activeDefect} />
            )}
            {activeRightTab === "quality" && (
              <QualityPanel report={report} defect={activeDefect} />
            )}
          </div>
        </div>
      </div>

      {/* Interactive 3D Metrology Fullscreen Modal */}
      <ThreeDModal
        isOpen={is3DModalOpen}
        onClose={() => setIs3DModalOpen(false)}
        inspectionId={inspectionId}
        defect={activeDefect}
      />
    </div>
  );
}
