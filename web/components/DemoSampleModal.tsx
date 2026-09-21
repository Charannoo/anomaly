"use client";

import React, { useEffect, useState } from "react";
import { X, Sparkles, Check, ArrowRight } from "lucide-react";
import { DemoCase, fetchDemoCases } from "@/lib/api";
import { StatusIndicator } from "./StatusIndicator";

interface DemoSampleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSample: (demoCase: DemoCase) => void;
}

export const DemoSampleModal: React.FC<DemoSampleModalProps> = ({
  isOpen,
  onClose,
  onSelectSample,
}) => {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchDemoCases()
        .then((data) => setCases(data))
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-3xl border border-border-default rounded-md bg-bg-panel shadow-panel flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border-default px-4 py-3 bg-bg-subtle">
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-accent-primary" />
            <div>
              <h3 className="text-sm font-semibold text-text-primary">
                Demo Sample Library
              </h3>
              <p className="text-[11px] text-text-muted">
                Pre-calibrated MVTec-3D multimodal benchmark samples
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-bg-surface transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {loading ? (
            <div className="p-8 text-center text-xs text-text-muted">Loading verified benchmark samples...</div>
          ) : (
            cases.map((c) => (
              <div
                key={c.id}
                onClick={() => {
                  onSelectSample(c);
                  onClose();
                }}
                className="group flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded border border-border-subtle hover:border-accent-primary bg-bg-subtle hover:bg-bg-surface transition-all cursor-pointer"
              >
                <div className="flex items-start sm:items-center gap-3">
                  <div className="w-12 h-12 rounded bg-bg-app border border-border-default overflow-hidden flex-shrink-0 flex items-center justify-center">
                    <img
                      src={c.thumbnail_url}
                      alt={c.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-text-primary group-hover:text-accent-primary transition-colors">
                        {c.name}
                      </span>
                      <StatusIndicator status={c.expected_decision} size="sm" />
                    </div>
                    <p className="text-[11px] text-text-muted mt-0.5">
                      {c.highlight}
                    </p>
                    <span className="inline-block mt-1 font-mono text-[10px] text-text-secondary bg-bg-app px-1.5 py-0.5 rounded border border-border-subtle">
                      {c.category} · {c.type}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2 self-end sm:self-center">
                  <span className="text-[11px] font-mono text-text-muted">
                    Score: <span className="text-text-primary">{c.expected_score}</span>
                  </span>
                  <div className="p-1 rounded bg-bg-app border border-border-subtle text-accent-primary group-hover:bg-accent-primary group-hover:text-bg-app transition-colors">
                    <ArrowRight size={13} />
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Modal Footer */}
        <div className="border-t border-border-default px-4 py-2.5 bg-bg-app flex justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1 rounded text-xs text-text-secondary hover:text-text-primary border border-border-default hover:bg-bg-surface transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default DemoSampleModal;
