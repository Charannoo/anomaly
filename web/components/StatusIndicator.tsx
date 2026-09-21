import React from "react";
import { cn } from "@/lib/utils";

interface StatusIndicatorProps {
  status: string;
  score?: number;
  size?: "sm" | "md" | "lg";
  className?: string;
  showText?: boolean;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  score,
  size = "md",
  className,
  showText = true,
}) => {
  const normStatus = (status || "").toUpperCase();
  const isAnomaly = normStatus.includes("ANOMAL") || normStatus.includes("DEFECT");
  const isReview = normStatus.includes("REVIEW") || normStatus.includes("MANUAL");
  const isNormal = normStatus === "NORMAL" || (!isAnomaly && !isReview);

  let dotColor = "bg-status-normal";
  let textColor = "text-status-normal";
  let label = "NORMAL";

  if (isReview) {
    dotColor = "bg-status-review";
    textColor = "text-status-review";
    label = "MANUAL REVIEW";
  } else if (isAnomaly) {
    dotColor = "bg-status-anomaly";
    textColor = "text-status-anomaly";
    label = "ANOMALY";
  }

  const dotSizes = {
    sm: "w-1.5 h-1.5",
    md: "w-2 h-2",
    lg: "w-2.5 h-2.5",
  };

  const textSizes = {
    sm: "text-[11px]",
    md: "text-[12px]",
    lg: "text-[13px]",
  };

  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <span className={cn("rounded-full flex-shrink-0", dotColor, dotSizes[size])} />
      {showText && (
        <span className={cn("font-medium tracking-wide font-mono", textColor, textSizes[size])}>
          {label}
        </span>
      )}
      {score !== undefined && (
        <span className="font-mono text-[11px] text-text-muted tabular-nums">
          ({score.toFixed(4)})
        </span>
      )}
    </div>
  );
};

export default StatusIndicator;
