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
    <div className="max-w-[1240px] mx-auto space-y-8">
      <div>
        <h1 className="text-[28px] font-semibold tracking-tight text-[#F3F5F7]">
          New Inspection
        </h1>
        <p className="text-xs text-[#A7AFBA] mt-0.5">
          Provide corresponding RGB and XYZ observations.
        </p>
      </div>

      <UploadPanel onRunInspection={handleRun} loading={loading} />
    </div>
  );
}
