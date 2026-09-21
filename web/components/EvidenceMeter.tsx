import React from "react";
import { cn } from "@/lib/utils";

interface EvidenceMeterProps {
  label: string;
  value: number | null | undefined;
  qualitativeLevel?: string;
  max?: number;
  highlight?: boolean;
  className?: string;
}

export const EvidenceMeter: React.FC<EvidenceMeterProps> = ({
  label,
  value,
  qualitativeLevel,
  max = 1.0,
  highlight = false,
  className,
}) => {
  const numVal = typeof value === "number" && !isNaN(value) ? value : 0;
  const pct = Math.min(100, Math.max(0, (numVal / max) * 100));

  let barColor = "bg-accent-primary";
  if (numVal > 0.8) {
    barColor = highlight ? "bg-status-anomaly" : "bg-accent-primary";
  } else if (numVal > 0.5) {
    barColor = "bg-status-review";
  }

  return (
    <div className={cn("space-y-1 text-xs", className)}>
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-text-secondary">{label}</span>
        <div className="flex items-center gap-1.5 font-mono">
          {qualitativeLevel && (
            <span className="text-text-muted text-[10px]">{qualitativeLevel}</span>
          )}
          <span className="text-text-primary font-medium tabular-nums">
            {typeof value === "number" ? value.toFixed(2) : "—"}
          </span>
        </div>
      </div>
      <div className="h-1.5 w-full rounded-full bg-bg-app border border-border-subtle overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-300", barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

export default EvidenceMeter;
