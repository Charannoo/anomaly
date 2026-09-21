"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Loader2, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

const STAGES = [
  "Input Validation",
  "RGB Feature Extraction",
  "XYZ Feature Extraction",
  "Normal Prototype Retrieval",
  "PNTC Scoring",
  "Defect Segmentation",
  "Geometry Analysis",
  "Volume Quantification",
  "Normal Twin Retrieval",
  "Evidence Trace",
  "Review Guard",
  "Report Generation",
];

interface PipelineTimelineProps {
  inspectionId: string;
}

export const PipelineTimeline: React.FC<PipelineTimelineProps> = ({ inspectionId }) => {
  const router = useRouter();
  const [currentStageIndex, setCurrentStageIndex] = useState(1);
  const [stageDescription, setStageDescription] = useState("Initializing inspection pipeline...");
  const [isDone, setIsDone] = useState(false);

  useEffect(() => {
    const eventSource = new EventSource(`/api/inspections/${inspectionId}/status`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setCurrentStageIndex(data.stage_index);
        setStageDescription(data.description);
        if (data.is_complete) {
          setIsDone(true);
          eventSource.close();
          setTimeout(() => {
            router.push(`/inspect/${inspectionId}`);
          }, 800);
        }
      } catch (err) {
        console.error("SSE parse error", err);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      // Auto fallback if SSE stream closes
      setIsDone(true);
      setTimeout(() => {
        router.push(`/inspect/${inspectionId}`);
      }, 1000);
    };

    return () => {
      eventSource.close();
    };
  }, [inspectionId, router]);

  return (
    <div className="space-y-6">
      {/* Central Architecture Schematic */}
      <div className="p-6 rounded-md border border-border-default bg-bg-panel flex flex-col items-center justify-center">
        <span className="text-[11px] font-mono uppercase tracking-wider text-text-muted mb-4">
          Pipeline Dataflow Schematic
        </span>
        <div className="font-mono text-xs text-text-secondary bg-bg-app p-4 rounded border border-border-subtle leading-relaxed text-center sm:text-left select-none">
          <div className="flex flex-col sm:flex-row items-center gap-2">
            <span className={cn(currentStageIndex >= 2 ? "text-accent-primary font-bold" : "text-text-muted")}>
              RGB Observation → DINOv2
            </span>
            <span className="text-text-muted">─┐</span>
          </div>
          <div className="pl-24 text-accent-primary font-semibold py-1">
            ├─► PNTC Multimodal Graph ─► Physical Metrology Analysis
          </div>
          <div className="flex flex-col sm:flex-row items-center gap-2">
            <span className={cn(currentStageIndex >= 3 ? "text-accent-primary font-bold" : "text-text-muted")}>
              XYZ Observation → Point-MAE
            </span>
            <span className="text-text-muted">─┘</span>
          </div>
        </div>
        <div className="mt-4 text-xs font-mono text-accent-primary flex items-center gap-2">
          {!isDone && <Loader2 size={13} className="animate-spin text-accent-primary" />}
          {isDone && <Check size={13} className="text-status-normal" />}
          <span>{stageDescription}</span>
        </div>
      </div>

      {/* Stage Timeline Grid */}
      <div className="border border-border-default rounded-md bg-bg-panel p-4">
        <div className="flex items-center justify-between border-b border-border-subtle pb-2.5 mb-4">
          <span className="text-xs font-mono font-semibold uppercase text-text-secondary">
            Pipeline Execution Stages
          </span>
          <span className="text-xs font-mono text-text-muted">
            Stage {Math.min(currentStageIndex, STAGES.length)} of {STAGES.length}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
          {STAGES.map((stageName, idx) => {
            const stageNum = idx + 1;
            const isPassed = currentStageIndex > stageNum || isDone;
            const isCurrent = currentStageIndex === stageNum && !isDone;

            return (
              <div
                key={stageName}
                className={cn(
                  "flex items-center gap-2.5 p-2 rounded border text-xs transition-colors font-mono",
                  isPassed
                    ? "border-border-subtle bg-bg-app text-text-primary"
                    : isCurrent
                    ? "border-accent-primary bg-bg-surface text-accent-primary font-semibold shadow-subtle"
                    : "border-border-subtle/50 bg-bg-subtle/40 text-text-muted"
                )}
              >
                <div
                  className={cn(
                    "w-5 h-5 rounded flex items-center justify-center flex-shrink-0 text-[10px]",
                    isPassed
                      ? "bg-status-normal-bg text-status-normal font-bold"
                      : isCurrent
                      ? "bg-accent-subtle text-accent-primary"
                      : "bg-bg-app text-text-muted"
                  )}
                >
                  {isPassed ? (
                    <Check size={11} />
                  ) : isCurrent ? (
                    <Loader2 size={11} className="animate-spin" />
                  ) : (
                    stageNum
                  )}
                </div>
                <span className="truncate">{stageName}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default PipelineTimeline;
