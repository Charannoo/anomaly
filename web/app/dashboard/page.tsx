"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { PlusCircle, ArrowRight } from "lucide-react";
import { InspectionTable } from "@/components/InspectionTable";
import { fetchAnalytics, fetchInspections, AnalyticsData, InspectionListItem } from "@/lib/api";

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [recentInspections, setRecentInspections] = useState<InspectionListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [aData, iData] = await Promise.all([
        fetchAnalytics(),
        fetchInspections({ limit: 10 }),
      ]);
      setAnalytics(aData);
      setRecentInspections(iData.items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="space-y-10 max-w-[1440px] mx-auto">
      {/* Clean Operational Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-4">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#F3F5F7]">
            PNTC Inspect
          </h1>
          <p className="text-xs text-[#A7AFBA] mt-0.5">
            Industrial RGB–3D Inspection
          </p>
        </div>

        <div>
          <Link
            href="/inspect"
            className="inline-flex items-center gap-2 px-4 py-2 rounded bg-[#5BB8C4] text-[#0B0D10] text-xs font-medium hover:bg-[#71C7D1] transition-colors"
          >
            <PlusCircle size={14} />
            <span>New Inspection</span>
          </Link>
        </div>
      </div>

      {/* Spacious Horizontal KPI Row (No outer card, subtle dividers) */}
      <div className="flex flex-wrap items-center divide-x divide-white/[0.06] py-1">
        <div className="pr-8 sm:pr-12">
          <div className="text-3xl font-semibold text-[#F3F5F7] tracking-tight tabular-nums">
            {analytics?.total_inspections ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Inspections</div>
        </div>

        <div className="px-8 sm:px-12">
          <div className="text-3xl font-semibold text-[#E96B6B] tracking-tight tabular-nums">
            {analytics?.anomalies ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Anomalies</div>
        </div>

        <div className="px-8 sm:px-12">
          <div className="text-3xl font-semibold text-[#55B98A] tracking-tight tabular-nums">
            {analytics?.normal ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Normal</div>
        </div>

        <div className="pl-8 sm:pl-12">
          <div className="text-3xl font-semibold text-[#D4A95B] tracking-tight tabular-nums">
            {analytics?.manual_review ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Review</div>
        </div>
      </div>

      <div className="border-t border-white/[0.06]" />

      {/* Main Content: Recent Inspections */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-[17px] font-semibold text-[#F3F5F7]">
            Recent inspections
          </h2>
          <Link
            href="/inspections"
            className="text-xs text-[#6F7884] hover:text-[#5BB8C4] transition-colors flex items-center gap-1"
          >
            <span>View all</span>
            <ArrowRight size={12} />
          </Link>
        </div>

        <InspectionTable inspections={recentInspections} />
      </div>
    </div>
  );
}
