"use client";

import React from "react";
import { useParams } from "next/navigation";
import { PipelineTimeline } from "@/components/PipelineTimeline";

export default function ProcessingPage() {
  const params = useParams();
  const inspectionId = params.id as string;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="pb-4 border-b border-border-default">
        <h1 className="text-xl font-semibold tracking-tight text-text-primary">
          Processing Inspection
        </h1>
        <p className="text-xs text-text-muted mt-0.5 font-mono">
          Task ID: {inspectionId} · Running frozen PNTC multimodal characterization
        </p>
      </div>

      <PipelineTimeline inspectionId={inspectionId} />
    </div>
  );
}
