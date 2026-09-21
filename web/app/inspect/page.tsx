"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { UploadPanel } from "@/components/UploadPanel";
import { runInspection } from "@/lib/api";

export default function NewInspectionPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleRun = async (payload: any) => {
    setLoading(true);
    try {
      const res = await runInspection(payload);
      router.push(`/inspect/${res.inspection_id}/processing`);
    } catch (e) {
      console.error(e);
      alert("Failed to execute inspection. Please check backend connection.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="pb-4 border-b border-border-default">
        <h1 className="text-xl font-semibold tracking-tight text-text-primary">
          New Inspection
        </h1>
        <p className="text-xs text-text-muted mt-0.5">
          Provide corresponding RGB and XYZ observations for automated PNTC multimodal characterization.
        </p>
      </div>

      <UploadPanel onRunInspection={handleRun} loading={loading} />
    </div>
  );
}
