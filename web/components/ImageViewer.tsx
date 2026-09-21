"use client";

import React, { useState } from "react";
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageViewerProps {
  inspectionId: string;
  defects?: any[];
  report?: any;
  selectedDefectId?: number;
  onSelectDefect?: (id: number) => void;
  onOpen3DView?: () => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  inspectionId,
  defects: propDefects,
  report,
  selectedDefectId = 1,
  onSelectDefect,
  onOpen3DView,
}) => {
  const defects = propDefects || report?.defects || [];
  const [activeTab, setActiveTab] = useState<string>("overlay");
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const tabs = [
    { id: "overlay", label: "Overlay" },
    { id: "rgb", label: "RGB" },
    { id: "heatmap", label: "Anomaly Map" },
    { id: "depth", label: "XYZ / Depth" },
    { id: "normal_twin", label: "Normal Twin" },
    { id: "surface_difference", label: "Difference" },
  ];

  const handleZoomIn = () => setZoomLevel((z) => Math.min(3, z + 0.25));
  const handleZoomOut = () => setZoomLevel((z) => Math.max(0.75, z - 0.25));
  const handleResetZoom = () => setZoomLevel(1);

  return (
    <div className="flex flex-col h-full border border-border-default rounded-md bg-bg-panel overflow-hidden">
      {/* Top Toolbar */}
      <div className="flex flex-wrap items-center justify-between border-b border-border-default px-3 py-2 bg-bg-subtle gap-2">
        {/* View Mode Tabs */}
        <div className="flex items-center gap-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition-colors whitespace-nowrap",
                activeTab === tab.id
                  ? "bg-bg-active text-accent-primary font-semibold border border-border-subtle"
                  : "text-text-muted hover:text-text-primary hover:bg-bg-surface"
              )}
            >
              {tab.label}
            </button>
          ))}
          {onOpen3DView && (
            <button
              onClick={onOpen3DView}
              className="px-2.5 py-1 rounded text-xs font-medium text-accent-primary hover:bg-accent-subtle transition-colors border border-accent-subtle whitespace-nowrap flex items-center gap-1"
            >
              <span>3D View</span>
              <Maximize2 size={11} />
            </button>
          )}
        </div>

        {/* Zoom & Fit Toolbar */}
        <div className="flex items-center gap-1 text-text-muted">
          <button
            onClick={handleZoomOut}
            className="p-1 rounded hover:text-text-primary hover:bg-bg-surface"
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
          <span className="text-[11px] font-mono px-1 w-10 text-center text-text-secondary">
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-1 rounded hover:text-text-primary hover:bg-bg-surface"
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1 rounded hover:text-text-primary hover:bg-bg-surface ml-1"
            title="Reset Zoom"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Main Visual Canvas Area */}
      <div className="flex-1 min-h-[380px] bg-bg-app relative flex items-center justify-center overflow-hidden p-4">
        <div
          className="transition-transform duration-150 flex items-center justify-center max-w-full max-h-full"
          style={{ transform: `scale(${zoomLevel})` }}
        >
          <img
            src={`/api/inspections/${inspectionId}/artifacts/${activeTab}`}
            alt={`${activeTab} view`}
            className="max-h-[500px] w-auto object-contain rounded select-none border border-border-subtle shadow-subtle"
            onError={(e) => {
              // Fallback to overlay if specific artifact not available
              (e.target as HTMLImageElement).src = `/api/inspections/${inspectionId}/artifacts/overlay`;
            }}
          />
        </div>

        {/* Defect Quick Select Pill Overlay (if multiple defects exist) */}
        {defects.length > 0 && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/75 backdrop-blur-sm px-2.5 py-1 rounded border border-border-default text-xs font-mono">
            <span className="text-text-muted text-[11px]">DEFECT:</span>
            {defects.map((d: any) => (
              <button
                key={d.id}
                onClick={() => onSelectDefect && onSelectDefect(d.id)}
                className={cn(
                  "px-1.5 py-0.5 rounded text-[11px] font-semibold transition-colors",
                  selectedDefectId === d.id
                    ? "bg-status-anomaly text-white"
                    : "text-text-secondary hover:text-text-primary hover:bg-white/10"
                )}
              >
                #{d.id}
              </button>
            ))}
          </div>
        )}

        {/* Subtle Heatmap Scale Legend */}
        {activeTab === "heatmap" || activeTab === "overlay" ? (
          <div className="absolute bottom-3 right-3 bg-black/75 backdrop-blur-sm px-3 py-1.5 rounded border border-border-default text-[11px] font-mono flex items-center gap-2">
            <span className="text-text-muted text-[10px]">NOMINAL</span>
            <div className="w-24 h-2 rounded-sm bg-gradient-to-r from-blue-900 via-amber-500 to-red-600" />
            <span className="text-status-anomaly text-[10px] font-bold">ANOMALY EVIDENCE</span>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default ImageViewer;
