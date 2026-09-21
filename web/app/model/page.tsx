"use client";

import React, { useEffect, useState } from "react";
import { fetchModelInfo } from "@/lib/api";
import { Cpu, Lock, CheckCircle2, AlertTriangle, Layers, GitFork, ShieldCheck } from "lucide-react";

export default function ModelExplorerPage() {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelInfo()
      .then((data) => setModelInfo(data))
      .catch((err) => console.error("Failed to load model info", err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
        {/* Top Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/[0.08] pb-5">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F7]">
                PNTC Architecture Explorer
              </h1>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-[#55B98A]/10 text-[#55B98A] border border-[#55B98A]/30">
                <Lock className="w-3 h-3" />
                CANONICAL FROZEN
              </span>
            </div>
            <p className="text-xs text-[#A7AFBA]">
              Paired Neighborhood Topology Consistency across multimodal RGB–3D representations.
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded bg-[#15191F] border border-white/[0.08] text-[#A7AFBA]">
              Tag: <span className="text-[#5BB8C4]">h5d-pntc-verified</span>
            </div>
            <div className="px-3 py-1.5 rounded bg-[#15191F] border border-white/[0.08] text-[#A7AFBA]">
              Status: <span className="text-[#55B98A]">Ready</span>
            </div>
          </div>
        </div>

        {/* Pipeline Architecture Schematic */}
        <div className="p-6 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#5BB8C4]" />
              <h2 className="text-sm font-semibold text-[#F3F5F7]">Information Flow & Fusion Topology</h2>
            </div>
            <span className="text-[11px] text-[#6F7884] font-mono">READ-ONLY SPECIFICATION</span>
          </div>

          <div className="p-5 rounded-md bg-[#0B0D10] border border-white/[0.06] overflow-x-auto">
            <div className="min-w-[640px] flex items-center justify-between text-xs font-mono">
              {/* RGB Branch */}
              <div className="flex flex-col gap-3 w-48">
                <div className="p-3 rounded border border-white/[0.12] bg-[#15191F] text-center">
                  <div className="text-[10px] text-[#6F7884]">MODALITY 1</div>
                  <div className="font-medium text-[#F3F5F7]">RGB Observation</div>
                  <div className="text-[10px] text-[#A7AFBA] mt-0.5">224 × 224 × 3</div>
                </div>
                <div className="text-center text-[#6F7884]">↓</div>
                <div className="p-3 rounded border border-[#5BB8C4]/40 bg-[#5BB8C4]/5 text-center">
                  <div className="text-[10px] text-[#5BB8C4]">BACKBONE</div>
                  <div className="font-medium text-[#F3F5F7]">DINOv2 ViT-B/14</div>
                  <div className="text-[10px] text-[#A7AFBA] mt-0.5">768-dim Patches</div>
                </div>
              </div>

              {/* Fusion Junction */}
              <div className="flex flex-col items-center justify-center px-6">
                <div className="text-[#6F7884] text-center text-[11px] mb-2 font-sans font-medium">
                  Paired Memory (15,000 Coreset)
                </div>
                <div className="p-4 rounded-lg border-2 border-[#5BB8C4] bg-[#191E25] shadow-lg text-center w-56">
                  <div className="text-[11px] font-bold text-[#5BB8C4] tracking-wider uppercase mb-1">
                    PNTC Fusion Core
                  </div>
                  <div className="text-[11px] text-[#F3F5F7] mb-1">Top-k Retrieval (k=5)</div>
                  <div className="text-[10px] text-[#A7AFBA]">JS Topology Divergence</div>
                  <div className="text-[10px] text-[#D4A95B] mt-1 font-semibold">λ = 0.35 + Bilateral Gate</div>
                </div>
                <div className="text-center text-[#6F7884] mt-2">↓</div>
                <div className="text-[10px] text-[#A7AFBA] mt-1">Image Aggregation (Mean top 0.5%)</div>
              </div>

              {/* XYZ Branch */}
              <div className="flex flex-col gap-3 w-48">
                <div className="p-3 rounded border border-white/[0.12] bg-[#15191F] text-center">
                  <div className="text-[10px] text-[#6F7884]">MODALITY 2</div>
                  <div className="font-medium text-[#F3F5F7]">XYZ Point Cloud</div>
                  <div className="text-[10px] text-[#A7AFBA] mt-0.5">Calibrated Organised (N, 3)</div>
                </div>
                <div className="text-center text-[#6F7884]">↓</div>
                <div className="p-3 rounded border border-[#5BB8C4]/40 bg-[#5BB8C4]/5 text-center">
                  <div className="text-[10px] text-[#5BB8C4]">BACKBONE</div>
                  <div className="font-medium text-[#F3F5F7]">Point-MAE</div>
                  <div className="text-[10px] text-[#A7AFBA] mt-0.5">1152-dim Embeddings</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Read-only Hyperparameters Grid */}
        <div className="p-6 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#F3F5F7]">Frozen Model Hyperparameters</h2>
            <span className="text-[11px] text-[#6F7884]">Immutable Evaluation Parameters</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[#A7AFBA] text-[11px] mb-1">RGB Backbone</div>
              <div className="text-[#F3F5F7] font-semibold">DINOv2 ViT-B/14</div>
              <div className="text-[10px] text-[#6F7884] mt-1">Dimension: 768</div>
            </div>

            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[#A7AFBA] text-[11px] mb-1">XYZ Backbone</div>
              <div className="text-[#F3F5F7] font-semibold">Point-MAE</div>
              <div className="text-[10px] text-[#6F7884] mt-1">Dimension: 1152</div>
            </div>

            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[#A7AFBA] text-[11px] mb-1">Prototype Coreset</div>
              <div className="text-[#5BB8C4] font-semibold">15,000 Paired</div>
              <div className="text-[10px] text-[#6F7884] mt-1">Greedy facility-location</div>
            </div>

            <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06]">
              <div className="text-[#A7AFBA] text-[11px] mb-1">Topology Weight (λ)</div>
              <div className="text-[#5BB8C4] font-semibold">0.3500</div>
              <div className="text-[10px] text-[#6F7884] mt-1">Neighborhood k: 5</div>
            </div>
          </div>
        </div>

        {/* Benchmark Comparison Table */}
        <div className="p-6 rounded-lg border border-white/[0.08] bg-[#15191F] space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-[#F3F5F7]">Benchmark Comparison (MVTec-3D AD)</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Comparison against recent published state-of-the-art multimodal industrial anomaly detection architectures.
            </p>
          </div>

          <div className="border border-white/[0.08] rounded-md overflow-hidden">
            <table className="w-full text-xs text-left">
              <thead className="bg-[#101318] text-[#A7AFBA] uppercase text-[10px] tracking-wider border-b border-white/[0.08]">
                <tr>
                  <th className="py-2.5 px-4">Architecture</th>
                  <th className="py-2.5 px-4">Provenance</th>
                  <th className="py-2.5 px-4 text-right">I-AUROC</th>
                  <th className="py-2.5 px-4 text-right">P-AUROC</th>
                  <th className="py-2.5 px-4 text-right">AUPRO@0.3</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04] font-mono">
                {modelInfo?.benchmark_comparison ? (
                  modelInfo.benchmark_comparison.map((row: any, i: number) => (
                    <tr
                      key={i}
                      className={row.is_current ? "bg-[#5BB8C4]/5 text-[#F3F5F7] font-semibold" : "hover:bg-white/[0.02] text-[#A7AFBA]"}
                    >
                      <td className="py-3 px-4 font-sans flex items-center gap-2">
                        {row.is_current && <ShieldCheck className="w-3.5 h-3.5 text-[#55B98A]" />}
                        {row.method}
                      </td>
                      <td className="py-3 px-4 text-[11px] text-[#6F7884] font-sans">{row.source}</td>
                      <td className="py-3 px-4 text-right tabular-nums text-[#F3F5F7]">
                        {(row.i_auroc * 100).toFixed(3)}%
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums text-[#F3F5F7]">
                        {(row.p_auroc * 100).toFixed(3)}%
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums text-[#F3F5F7]">
                        {(row.aupro * 100).toFixed(3)}%
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="py-4 text-center text-[#6F7884]">
                      Loading benchmark verification data...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="p-3.5 rounded bg-[#101318] border border-white/[0.06] text-xs text-[#A7AFBA] flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-[#D4A95B] shrink-0 mt-0.5" />
            <div>
              <span className="font-medium text-[#F3F5F7]">Dataset Manifest & Protocol Caveat:</span> Prior work
              reported figures correspond to official published papers under standard MVTec-3D protocols. PNTC
              measurements were computed on the frozen canonical verification split (10 categories, seed=42) and verified
              with zero post-hoc test tuning.
            </div>
          </div>
        </div>
      </div>
  );
}
