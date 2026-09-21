"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Download, Cuboid, ChevronRight } from "lucide-react";
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
      <div className="flex flex-col items-center justify-center p-24 text-text-muted text-xs">
        Loading inspection data...
      </div>
    );
  }

  if (!inspection) {
    return (
      <div className="max-w-xl mx-auto p-12 text-center border border-white/[0.06] rounded bg-bg-surface space-y-3">
        <h2 className="text-base font-semibold text-text-primary">Inspection not found</h2>
        <p className="text-xs text-text-muted">The requested inspection record could not be resolved from local persistence.</p>
        <Link href="/inspections" className="inline-block text-xs text-accent-primary hover:underline">
          ← Back to inspections
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
    <div className="space-y-5 max-w-[1920px] mx-auto min-h-[calc(100vh-100px)] flex flex-col">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-4 border-b border-white/[0.05] flex-shrink-0">
        <div>
          <div className="flex items-center gap-1.5 text-xs text-text-muted mb-1">
            <Link href="/inspections" className="hover:text-text-primary transition-colors">
              Inspections
            </Link>
            <ChevronRight size={12} />
            <span className="text-text-secondary capitalize">{inspection.category}</span>
            <ChevronRight size={12} />
            <span className="text-text-primary font-mono">{inspection.id}</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-text-primary flex items-center gap-2">
            <span>Inspection {inspection.id}</span>
            <span className="text-xs font-normal text-text-muted">
              ({inspection.sample_id})
            </span>
          </h1>
        </div>

        {/* Top Action Toolbar */}
        <div className="flex items-center gap-2.5 relative">
          <button
            onClick={() => setIs3DModalOpen(true)}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-bg-surface hover:bg-bg-elevated border border-white/[0.08] text-text-primary text-xs font-medium transition-colors"
          >
            <Cuboid size={14} className="text-accent-primary" />
            <span>3D metrology view</span>
          </button>

          {/* Export Dropdown Menu */}
          <div className="relative">
            <button
              onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-bg-surface hover:bg-bg-elevated border border-white/[0.08] text-text-primary text-xs font-medium transition-colors"
            >
              <Download size={14} />
              <span>Export</span>
            </button>

            {isExportMenuOpen && (
              <div className="absolute right-0 mt-1 w-52 border border-white/[0.08] rounded bg-bg-surface shadow-xl py-1 z-50 text-xs">
                <a
                  href={`/api/inspections/${inspectionId}/export/html`}
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3.5 py-2 hover:bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
                >
                  Printable ISO HTML / PDF
                </a>
                <a
                  href={`/api/inspections/${inspectionId}/export/json`}
                  download
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3.5 py-2 hover:bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
                >
                  Machine JSON output
                </a>
                <a
                  href={`/api/inspections/${inspectionId}/export/text`}
                  download
                  onClick={() => setIsExportMenuOpen(false)}
                  className="block px-3.5 py-2 hover:bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
                >
                  Technical summary (TXT)
                </a>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quiet Metric Strip - No Heavy Enclosure */}
      <div className="flex flex-wrap items-center justify-between gap-4 py-2 border-b border-white/[0.04] text-xs flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                isReview ? "bg-status-review" : isAnomaly ? "bg-status-anomaly" : "bg-status-normal"
              )}
            />
            <span
              className={cn(
                "font-semibold",
                isReview ? "text-status-review" : isAnomaly ? "text-status-anomaly" : "text-status-normal"
              )}
            >
              {isReview ? "Manual review" : isAnomaly ? "Defect detected" : "Normal"}
            </span>
          </div>
          <span className="text-white/[0.1]">|</span>
          <span className="text-text-secondary">
            Score:{" "}
            <strong className="text-text-primary tabular-nums font-mono">
              {formatMetric(pntc.score, 4)}
            </strong>
          </span>
          <span className="text-white/[0.1]">|</span>
          <span className="text-text-secondary">
            Threshold:{" "}
            <span className="text-text-primary tabular-nums font-mono">
              {pntc.threshold || 0.5}
            </span>
          </span>
        </div>

        <div className="flex items-center gap-5 text-text-muted">
          <div>
            Certainty: <strong className="text-text-primary font-medium">{report.decision_certainty || "High"}</strong>
          </div>
          <div>
            Regions:{" "}
            <strong className="text-text-primary tabular-nums font-mono">{defects.length}</strong>
          </div>
          <div>
            Execution:{" "}
            <strong className="text-text-primary font-mono">{formatMetric(inspection.execution_time_ms, 1)} ms</strong>
          </div>
        </div>
      </div>

      {/* Main Three-Column Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0 overflow-hidden pt-2">
        {/* LEFT COLUMN: Visual Inspection (5 cols) */}
        <div className="lg:col-span-5 h-full overflow-hidden flex flex-col">
          <ImageViewer
            inspectionId={inspectionId}
            defects={defects}
            selectedDefectId={selectedDefectId}
            onSelectDefect={(id) => setSelectedDefectId(id)}
            onOpen3DView={() => setIs3DModalOpen(true)}
          />
        </div>

        {/* CENTER COLUMN: Defect Inspector (4 cols) */}
        <div className="lg:col-span-4 h-full overflow-y-auto">
          <DefectInspector
            defect={activeDefect}
            overallDecision={decision}
            onViewTrace={() => setActiveRightTab("trace")}
          />
        </div>

        {/* RIGHT COLUMN: Inspector Drawer Context (3 cols) */}
        <div className="lg:col-span-3 h-full border border-white/[0.06] rounded-md bg-bg-surface flex flex-col overflow-hidden">
          {/* Drawer Tabs */}
          <div className="flex items-center border-b border-white/[0.05] bg-bg-subtle px-3 py-2 gap-1 text-xs overflow-x-auto flex-shrink-0">
            <button
              onClick={() => setActiveRightTab("explain")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors whitespace-nowrap text-xs",
                activeRightTab === "explain"
                  ? "bg-bg-elevated text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Explain
            </button>
            <button
              onClick={() => setActiveRightTab("normal_twin")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors whitespace-nowrap text-xs",
                activeRightTab === "normal_twin"
                  ? "bg-bg-elevated text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Normal twin
            </button>
            <button
              onClick={() => setActiveRightTab("trace")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors whitespace-nowrap text-xs",
                activeRightTab === "trace"
                  ? "bg-bg-elevated text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Trace
            </button>
            <button
              onClick={() => setActiveRightTab("quality")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors whitespace-nowrap text-xs",
                activeRightTab === "quality"
                  ? "bg-bg-elevated text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Quality
            </button>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 p-4 overflow-y-auto">
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
