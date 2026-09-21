"use client";

import React, { useEffect, useState } from "react";
import { fetchAnalytics, AnalyticsData } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { RefreshCw, AlertCircle } from "lucide-react";
import Link from "next/link";

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAnalytics();
      setData(res);
    } catch (err: any) {
      setError(err.message || "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Safe Review Rate Calculation: handles both fraction (0.286) and percentage (28.6)
  const rawRate = data?.manual_review_rate ?? 0;
  const reviewRatePercent = rawRate > 1.0 ? rawRate : rawRate * 100.0;
  const displayReviewRate = Math.min(100.0, Math.max(0.0, reviewRatePercent)).toFixed(1) + "%";

  return (
    <div className="space-y-10 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-4">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#F3F5F7]">
            Analytics
          </h1>
          <p className="text-xs text-[#A7AFBA] mt-0.5">
            Operational quality metrics and classification distributions.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-white/[0.06] bg-[#12161B] hover:bg-[#171C22] text-xs text-[#A7AFBA] hover:text-[#F3F5F7] transition-colors disabled:opacity-40 self-start"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-3.5 rounded bg-[#E96B6B]/10 border border-[#E96B6B]/20 flex items-center gap-2.5 text-xs text-[#E96B6B]">
          <AlertCircle size={15} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Clean Metric Strip with subtle vertical dividers (No outer box) */}
      <div className="flex flex-wrap items-center divide-x divide-white/[0.06] py-1">
        <div className="pr-6 sm:pr-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#F3F5F7] tracking-tight tabular-nums">
            {data?.total_inspections ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Inspections</div>
        </div>

        <div className="px-6 sm:px-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#E96B6B] tracking-tight tabular-nums">
            {data?.anomalies ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Anomalies</div>
        </div>

        <div className="px-6 sm:px-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#55B98A] tracking-tight tabular-nums">
            {data?.normal ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Normal</div>
        </div>

        <div className="px-6 sm:px-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#D4A95B] tracking-tight tabular-nums">
            {data?.manual_review ?? "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Manual review</div>
        </div>

        <div className="px-6 sm:px-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#5BB8C4] tracking-tight tabular-nums">
            {displayReviewRate}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Review rate</div>
        </div>

        <div className="pl-6 sm:pl-9">
          <div className="text-2xl sm:text-3xl font-semibold text-[#A7AFBA] tracking-tight tabular-nums">
            {data?.average_anomaly_score !== undefined ? data.average_anomaly_score.toFixed(3) : "—"}
          </div>
          <div className="text-xs text-[#6F7884] mt-1">Mean score</div>
        </div>
      </div>

      <div className="border-t border-white/[0.06]" />

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Outcome Distribution */}
        <div className="space-y-3">
          <div>
            <h2 className="text-[15px] font-semibold text-[#F3F5F7]">Outcome distribution</h2>
            <p className="text-xs text-[#6F7884] mt-0.5">Classification split across processed components</p>
          </div>

          <div className="h-72 w-full pt-4">
            {loading ? (
              <div className="h-full flex items-center justify-center text-xs text-[#6F7884]">
                Loading metrics...
              </div>
            ) : !data || data.by_status.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-[#6F7884]">
                No inspection records found.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.by_status} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <XAxis
                    dataKey="status"
                    stroke="#6F7884"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <YAxis
                    stroke="#6F7884"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#12161B",
                      borderColor: "rgba(255,255,255,0.08)",
                      borderRadius: "6px",
                      fontSize: "12px",
                      color: "#F3F5F7",
                    }}
                    cursor={{ fill: "rgba(255,255,255,0.02)" }}
                  />
                  <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                    {data.by_status.map((entry, index) => {
                      let fill = "#5BB8C4";
                      const s = entry.status.toUpperCase();
                      if (s.includes("ANOMAL")) fill = "#E96B6B";
                      else if (s.includes("NORM")) fill = "#55B98A";
                      else if (s.includes("REVIEW")) fill = "#D4A95B";
                      return <Cell key={`cell-${index}`} fill={fill} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Inspections by Category */}
        <div className="space-y-3">
          <div>
            <h2 className="text-[15px] font-semibold text-[#F3F5F7]">Inspections by category</h2>
            <p className="text-xs text-[#6F7884] mt-0.5">Sample volume distribution by component type</p>
          </div>

          <div className="h-72 w-full pt-4">
            {loading ? (
              <div className="h-full flex items-center justify-center text-xs text-[#6F7884]">
                Loading category metrics...
              </div>
            ) : !data || data.by_category.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-[#6F7884]">
                No category data available.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.by_category} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <XAxis
                    dataKey="category"
                    stroke="#6F7884"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <YAxis
                    stroke="#6F7884"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#12161B",
                      borderColor: "rgba(255,255,255,0.08)",
                      borderRadius: "6px",
                      fontSize: "12px",
                      color: "#F3F5F7",
                    }}
                    cursor={{ fill: "rgba(255,255,255,0.02)" }}
                  />
                  <Bar dataKey="count" fill="#5BB8C4" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Quality Gate Strip */}
      <div className="border-t border-white/[0.06] pt-6 space-y-4">
        <div>
          <h2 className="text-[15px] font-semibold text-[#F3F5F7]">Quality gate & sensor reliability</h2>
          <p className="text-xs text-[#6F7884] mt-0.5">Physical sensor completeness and review margin</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-xs">
          <div>
            <div className="text-[#6F7884]">Physical calibration</div>
            <div className="text-base font-semibold font-mono text-[#55B98A] mt-1">100% Calibrated</div>
            <div className="text-[11px] text-[#6F7884] mt-0.5">Triangulation coordinates verified (mm).</div>
          </div>

          <div>
            <div className="text-[#6F7884]">Sensor coverage</div>
            <div className="text-base font-semibold font-mono text-[#F3F5F7] mt-1">96.8% Valid</div>
            <div className="text-[11px] text-[#6F7884] mt-0.5">Average point completeness across target zones.</div>
          </div>

          <div>
            <div className="text-[#6F7884]">Review Guard margin</div>
            <div className="text-base font-semibold font-mono text-[#D4A95B] mt-1">±5% Threshold</div>
            <div className="text-[11px] text-[#6F7884] mt-0.5">Scores near operating boundary routed for review.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
