"use client";

import React from "react";
import Link from "next/link";
import { ChevronRight, ArrowUpRight } from "lucide-react";
import { StatusIndicator } from "./StatusIndicator";
import { InspectionListItem } from "@/lib/api";
import { formatMetric } from "@/lib/utils";

interface InspectionTableProps {
  inspections: InspectionListItem[];
  showPagination?: boolean;
}

export const InspectionTable: React.FC<InspectionTableProps> = ({ inspections }) => {
  if (!inspections || inspections.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center border border-border-default rounded-lg bg-bg-panel">
        <span className="text-sm font-medium text-text-primary mb-1">No inspections recorded</span>
        <span className="text-xs text-text-muted mb-4">Execute your first RGB–3D inspection to populate this workspace.</span>
        <Link
          href="/inspect"
          className="px-3.5 py-1.5 rounded-DEFAULT bg-accent-primary text-bg-app text-xs font-semibold hover:bg-accent-hover transition-colors"
        >
          New Inspection
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto border border-border-default rounded-md bg-bg-panel">
      <table className="w-full text-left text-xs">
        <thead className="bg-bg-subtle border-b border-border-default text-text-secondary uppercase tracking-wider text-[11px] font-medium font-mono">
          <tr>
            <th className="py-2.5 px-3">ID</th>
            <th className="py-2.5 px-3">Sample</th>
            <th className="py-2.5 px-3">Category</th>
            <th className="py-2.5 px-3">Status</th>
            <th className="py-2.5 px-3 text-right">Score</th>
            <th className="py-2.5 px-3 text-center">Defects</th>
            <th className="py-2.5 px-3">Review State</th>
            <th className="py-2.5 px-3 text-right">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle font-sans">
          {inspections.map((insp) => (
            <tr
              key={insp.id}
              className="hover:bg-bg-surface transition-colors cursor-pointer group"
            >
              <td className="py-2.5 px-3 font-mono text-[12px] font-medium text-accent-primary whitespace-nowrap">
                <Link href={`/inspect/${insp.id}`} className="hover:underline">
                  {insp.id}
                </Link>
              </td>
              <td className="py-2.5 px-3 font-mono text-text-primary max-w-[180px] truncate" title={insp.sample_id}>
                {insp.sample_id}
              </td>
              <td className="py-2.5 px-3 text-text-secondary capitalize">
                {insp.category}
              </td>
              <td className="py-2.5 px-3 whitespace-nowrap">
                <StatusIndicator status={insp.status} size="sm" />
              </td>
              <td className="py-2.5 px-3 text-right font-mono tabular-nums font-medium text-text-primary">
                {formatMetric(insp.anomaly_score, 4)}
              </td>
              <td className="py-2.5 px-3 text-center font-mono tabular-nums">
                {insp.num_defects > 0 ? (
                  <span className="px-1.5 py-0.5 rounded bg-status-anomaly-bg text-status-anomaly text-[11px] font-semibold">
                    {insp.num_defects}
                  </span>
                ) : (
                  <span className="text-text-muted">0</span>
                )}
              </td>
              <td className="py-2.5 px-3 text-text-muted whitespace-nowrap">
                {insp.manual_review_recommended ? (
                  <span className="text-status-review font-medium text-[11px]">Flagged</span>
                ) : (
                  <span className="text-status-normal text-[11px]">Pass</span>
                )}
              </td>
              <td className="py-2.5 px-3 text-right whitespace-nowrap">
                <Link
                  href={`/inspect/${insp.id}`}
                  className="inline-flex items-center gap-1 text-[11px] text-text-secondary group-hover:text-accent-primary font-medium transition-colors"
                >
                  View
                  <ChevronRight size={13} />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default InspectionTable;
