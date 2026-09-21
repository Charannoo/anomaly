"use client";

import React from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
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
      <div className="flex flex-col items-center justify-center p-12 text-center border border-white/[0.04] rounded-lg bg-[#0E1115]">
        <span className="text-sm font-medium text-[#F3F5F7] mb-1">No inspections recorded</span>
        <span className="text-xs text-[#6F7884] mb-4">Execute your first RGB–3D inspection to populate this workspace.</span>
        <Link
          href="/inspect"
          className="px-3.5 py-1.5 rounded bg-[#5BB8C4] text-[#0B0D10] text-xs font-medium hover:bg-[#71C7D1] transition-colors"
        >
          New Inspection
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full text-left text-[13.5px]">
        <thead className="border-b border-white/[0.06] text-[#6F7884] text-xs font-normal">
          <tr>
            <th className="py-2.5 px-3 font-normal">ID</th>
            <th className="py-2.5 px-3 font-normal">Sample</th>
            <th className="py-2.5 px-3 font-normal">Category</th>
            <th className="py-2.5 px-3 font-normal">Status</th>
            <th className="py-2.5 px-3 text-right font-normal">Score</th>
            <th className="py-2.5 px-3 text-center font-normal">Defects</th>
            <th className="py-2.5 px-3 font-normal">Review</th>
            <th className="py-2.5 px-3 text-right font-normal">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.03]">
          {inspections.map((insp) => (
            <tr
              key={insp.id}
              className="hover:bg-[#12161B] transition-colors cursor-pointer group"
            >
              <td className="py-3.5 px-3 font-mono text-xs font-medium text-[#5BB8C4] whitespace-nowrap">
                <Link href={`/inspect/${insp.id}`} className="hover:underline">
                  {insp.id}
                </Link>
              </td>
              <td className="py-3.5 px-3 font-mono text-xs text-[#F3F5F7] max-w-[200px] truncate" title={insp.sample_id}>
                {insp.sample_id}
              </td>
              <td className="py-3.5 px-3 text-[#A7AFBA] capitalize text-xs">
                {insp.category}
              </td>
              <td className="py-3.5 px-3 whitespace-nowrap text-xs">
                <StatusIndicator status={insp.status} size="sm" />
              </td>
              <td className="py-3.5 px-3 text-right font-mono tabular-nums text-xs font-medium text-[#F3F5F7]">
                {formatMetric(insp.anomaly_score, 4)}
              </td>
              <td className="py-3.5 px-3 text-center font-mono tabular-nums text-xs">
                {insp.num_defects > 0 ? (
                  <span className="text-[#E96B6B] font-medium">
                    {insp.num_defects}
                  </span>
                ) : (
                  <span className="text-[#6F7884]">—</span>
                )}
              </td>
              <td className="py-3.5 px-3 text-xs whitespace-nowrap">
                {insp.manual_review_recommended ? (
                  <span className="text-[#D4A95B] font-medium">Flagged</span>
                ) : (
                  <span className="text-[#6F7884]">Pass</span>
                )}
              </td>
              <td className="py-3.5 px-3 text-right whitespace-nowrap text-xs">
                <Link
                  href={`/inspect/${insp.id}`}
                  className="inline-flex items-center gap-1 text-[#6F7884] group-hover:text-[#5BB8C4] transition-colors"
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
