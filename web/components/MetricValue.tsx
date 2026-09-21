import React from "react";
import { cn } from "@/lib/utils";

interface MetricValueProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  className?: string;
  trend?: string;
  status?: "normal" | "anomaly" | "warning" | string;
  precision?: number;
  help?: string;
}

export const MetricValue: React.FC<MetricValueProps> = ({
  label,
  value,
  unit,
  subtext,
  className,
  trend,
  status,
  precision,
  help,
}) => {
  let displayVal = value;
  if (typeof value === "number" && precision !== undefined) {
    displayVal = value.toFixed(precision);
  }

  let valColor = "text-text-primary";
  if (status === "anomaly") valColor = "text-status-anomaly";
  if (status === "normal") valColor = "text-status-normal";
  if (status === "warning") valColor = "text-status-warning";

  return (
    <div className={cn("flex flex-col p-3 rounded bg-[#101318] border border-white/[0.06]", className)}>
      <span className="text-[11px] font-medium tracking-wide uppercase text-text-muted mb-1 truncate" title={label}>
        {label}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span className={cn("text-xl font-semibold tracking-tight font-mono tabular-nums", valColor)}>
          {displayVal}
        </span>
        {unit && (
          <span className="text-xs text-text-muted font-mono font-normal">
            {unit}
          </span>
        )}
        {trend && (
          <span className="text-[11px] font-mono text-status-normal ml-1">
            {trend}
          </span>
        )}
      </div>
      {(subtext || help) && (
        <span className="text-[10px] text-text-secondary mt-0.5 truncate" title={subtext || help}>
          {subtext || help}
        </span>
      )}
    </div>
  );
};

export default MetricValue;
