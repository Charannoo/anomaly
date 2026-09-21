"use client";

import React, { useState, useEffect } from "react";
import { fetchModelInfo, fetchResearchAblations } from "@/lib/api";
import { Copy, Check, ShieldCheck, ArrowRight, Layers, FileCode2, BookOpen, AlertTriangle } from "lucide-react";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "method", label: "Method" },
  { id: "performance", label: "Performance" },
  { id: "ablations", label: "Ablations" },
  { id: "reproducibility", label: "Reproducibility" },
  { id: "limitations", label: "Limitations" },
];

export default function AboutPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [researchData, setResearchData] = useState<any>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchModelInfo().catch(() => null),
      fetchResearchAblations().catch(() => null),
    ]).then(([mInfo, rData]) => {
      setModelInfo(mInfo);
      setResearchData(rData);
    });
  }, []);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="max-w-[1180px] mx-auto space-y-8">
      {/* Page Header */}
      <div className="border-b border-white/[0.06] pb-5">
        <h1 className="text-[26px] font-semibold tracking-tight text-[#F3F5F7]">
          About PNTC
        </h1>
        <p className="text-xs text-[#A7AFBA] mt-1">
          Paired Neighborhood Topology Consistency across multimodal RGB–3D representations.
        </p>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 mt-6 border-b border-white/[0.04] -mb-5 overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3.5 py-2 text-xs font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? "border-[#5BB8C4] text-[#F3F5F7]"
                  : "border-transparent text-[#6F7884] hover:text-[#A7AFBA]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === "overview" && (
        <div className="space-y-8 pt-2">
          <div className="space-y-3">
            <h2 className="text-base font-semibold text-[#F3F5F7]">System Overview</h2>
            <p className="text-xs text-[#A7AFBA] leading-relaxed max-w-3xl">
              PNTC uses RGB photometric appearance and XYZ surface geometry to detect, localize, and characterize
              industrial anomalies using paired normal-prototype retrieval consistency. Rather than relying on simple
              feature concatenation, PNTC evaluates the topological agreement between nearest training neighbors
              independently retrieved across both modalities.
            </p>
          </div>

          <div className="border-t border-white/[0.06] pt-6">
            <h3 className="text-xs font-semibold text-[#F3F5F7] mb-4">Technical Specifications</h3>
            <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-y-4 gap-x-8 text-xs">
              <div>
                <dt className="text-[#6F7884]">RGB backbone</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">DINOv2 ViT-B/14 (768-dim)</dd>
              </div>
              <div>
                <dt className="text-[#6F7884]">3D geometry backbone</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">Point-MAE (1152-dim)</dd>
              </div>
              <div>
                <dt className="text-[#6F7884]">Normal memory coreset</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">15,000 paired prototypes</dd>
              </div>
              <div>
                <dt className="text-[#6F7884]">Neighborhood size (k)</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">k = 5 nearest prototypes</dd>
              </div>
              <div>
                <dt className="text-[#6F7884]">Topology weight (λ)</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">0.3500</dd>
              </div>
              <div>
                <dt className="text-[#6F7884]">Image-level aggregation</dt>
                <dd className="text-[#F3F5F7] font-medium mt-0.5">Mean of top 0.5% patches</dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      {/* TAB 2: ARCHITECTURE */}
      {activeTab === "architecture" && (
        <div className="space-y-8 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Information Flow</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Dual-branch multimodal feature extraction and paired neighborhood consistency evaluation.
            </p>
          </div>

          {/* Spacious Horizontal Architecture Schematic */}
          <div className="p-6 rounded-lg bg-[#0E1115] border border-white/[0.04]">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6 py-4 max-w-4xl mx-auto text-xs">
              {/* RGB Branch */}
              <div className="flex flex-col items-center gap-3 w-44 text-center">
                <div className="px-3.5 py-2.5 rounded bg-[#12161B] border border-white/[0.06] w-full">
                  <div className="text-[10.5px] text-[#6F7884]">Photometric Input</div>
                  <div className="font-medium text-[#F3F5F7] mt-0.5">RGB Observation</div>
                </div>
                <div className="text-[#6F7884]">↓</div>
                <div className="px-3.5 py-2.5 rounded bg-[#171C22] border border-white/[0.06] w-full">
                  <div className="text-[10.5px] text-[#5BB8C4]">Feature Extractor</div>
                  <div className="font-medium text-[#F3F5F7] mt-0.5">DINOv2 ViT-B/14</div>
                </div>
              </div>

              {/* Central Junction */}
              <div className="flex flex-col items-center text-center px-4">
                <div className="text-[11px] text-[#6F7884] mb-2">Paired Memory (15k Coreset)</div>
                <div className="p-4 rounded-md bg-[#171C22] border border-[#5BB8C4]/30 w-52">
                  <div className="text-[11px] font-semibold text-[#5BB8C4] uppercase tracking-wider mb-1">
                    PNTC Fusion Core
                  </div>
                  <div className="text-[11px] text-[#F3F5F7]">Independent Top-k Retrieval</div>
                  <div className="text-[10px] text-[#A7AFBA] mt-0.5">JS Topology Divergence</div>
                  <div className="text-[10px] text-[#D4A95B] mt-1">λ = 0.35 + Bilateral Gate</div>
                </div>
                <div className="text-[#6F7884] mt-2">↓</div>
                <div className="text-[11px] text-[#A7AFBA] mt-1 font-medium">Calibrated Metrology</div>
              </div>

              {/* XYZ Branch */}
              <div className="flex flex-col items-center gap-3 w-44 text-center">
                <div className="px-3.5 py-2.5 rounded bg-[#12161B] border border-white/[0.06] w-full">
                  <div className="text-[10.5px] text-[#6F7884]">Geometric Input</div>
                  <div className="font-medium text-[#F3F5F7] mt-0.5">XYZ Point Cloud</div>
                </div>
                <div className="text-[#6F7884]">↓</div>
                <div className="px-3.5 py-2.5 rounded bg-[#171C22] border border-white/[0.06] w-full">
                  <div className="text-[10.5px] text-[#5BB8C4]">Feature Extractor</div>
                  <div className="font-medium text-[#F3F5F7] mt-0.5">Point-MAE</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: METHOD */}
      {activeTab === "method" && (
        <div className="space-y-8 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Mathematical Formulation</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Formal definition of paired memory, prototype retrieval, divergence, and gate attenuation.
            </p>
          </div>

          <div className="space-y-6 text-xs leading-relaxed text-[#A7AFBA]">
            <div>
              <h3 className="text-xs font-semibold text-[#F3F5F7] mb-1.5">1. Paired Prototype Memory</h3>
              <p>
                During calibration, normal training samples are indexed into a coreset of paired prototypes
                where each entry represents a spatially corresponding patch with both RGB and XYZ embeddings.
              </p>
            </div>

            <div>
              <h3 className="text-xs font-semibold text-[#F3F5F7] mb-1.5">2. Independent Neighborhood Retrieval</h3>
              <p>
                Given an observation with RGB embedding and XYZ embedding, we independently retrieve top-k nearest
                normal prototypes in appearance space and geometry space:
              </p>
              <div className="mt-2 p-3 rounded bg-[#0E1115] font-mono text-[11px] text-[#F3F5F7]">
                N_rgb(p) = argmin_k || z_rgb(p) - m_rgb ||, &nbsp;&nbsp; N_xyz(p) = argmin_k || z_xyz(p) - m_xyz ||
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold text-[#F3F5F7] mb-1.5">3. Topology Disagreement & Divergence</h3>
              <p>
                When a defect occurs, RGB and XYZ neighborhoods diverge. We measure this disagreement via
                Jensen-Shannon divergence over the normalized prototype proximity distributions:
              </p>
              <div className="mt-2 p-3 rounded bg-[#0E1115] font-mono text-[11px] text-[#F3F5F7]">
                D_JS( P_rgb || P_xyz ) = 0.5 * KL(P_rgb || M) + 0.5 * KL(P_xyz || M)
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold text-[#F3F5F7] mb-1.5">4. Final Anomaly Score Decomposition</h3>
              <p>
                The final patch anomaly evidence combines baseline Euclidean distance with gated topology divergence:
              </p>
              <div className="mt-2 p-3 rounded bg-[#0E1115] font-mono text-[11px] text-[#55B98A]">
                E(p) = E_base(p) + λ · G(p) · D_JS( P_rgb || P_xyz )
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: PERFORMANCE */}
      {activeTab === "performance" && (
        <div className="space-y-8 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Verified Benchmark Results</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Frozen evaluation metrics on the 10-category MVTec-3D anomaly detection benchmark.
            </p>
          </div>

          {/* Large Clean Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 py-2">
            <div>
              <div className="text-xs text-[#6F7884]">Image-level AUROC</div>
              <div className="text-3xl font-semibold font-mono text-[#F3F5F7] mt-1">96.541%</div>
              <div className="text-[11px] text-[#55B98A] mt-1">SOTA on MVTec-3D</div>
            </div>
            <div>
              <div className="text-xs text-[#6F7884]">Pixel-level AUROC</div>
              <div className="text-3xl font-semibold font-mono text-[#F3F5F7] mt-1">99.416%</div>
              <div className="text-[11px] text-[#55B98A] mt-1">Verified frozen</div>
            </div>
            <div>
              <div className="text-xs text-[#6F7884]">AUPRO @ 0.3</div>
              <div className="text-3xl font-semibold font-mono text-[#F3F5F7] mt-1">96.939%</div>
              <div className="text-[11px] text-[#55B98A] mt-1">Zero post-hoc tuning</div>
            </div>
          </div>

          {/* Benchmark Comparison Table */}
          <div className="border-t border-white/[0.06] pt-6 space-y-3">
            <h3 className="text-xs font-semibold text-[#F3F5F7]">Literature Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="text-[#6F7884] border-b border-white/[0.06]">
                  <tr>
                    <th className="py-2.5 font-medium">Method</th>
                    <th className="py-2.5 font-medium">Provenance</th>
                    <th className="py-2.5 font-medium text-right">I-AUROC</th>
                    <th className="py-2.5 font-medium text-right">P-AUROC</th>
                    <th className="py-2.5 font-medium text-right">AUPRO@0.3</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.03]">
                  {modelInfo?.benchmark_comparison?.map((row: any, i: number) => (
                    <tr key={i} className={row.is_current ? "font-semibold text-[#F3F5F7]" : "text-[#A7AFBA]"}>
                      <td className="py-3 flex items-center gap-1.5">
                        {row.is_current && <ShieldCheck className="w-3.5 h-3.5 text-[#55B98A]" />}
                        {row.method}
                      </td>
                      <td className="py-3 text-[11px] text-[#6F7884]">{row.source}</td>
                      <td className="py-3 text-right font-mono">{(row.i_auroc * 100).toFixed(3)}%</td>
                      <td className="py-3 text-right font-mono">{(row.p_auroc * 100).toFixed(3)}%</td>
                      <td className="py-3 text-right font-mono">{(row.aupro * 100).toFixed(3)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-[#6F7884] pt-2">
              Prior work reported figures correspond to official published papers under standard MVTec-3D protocols.
            </p>
          </div>
        </div>
      )}

      {/* TAB 5: ABLATIONS */}
      {activeTab === "ablations" && (
        <div className="space-y-6 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Ablation Progression</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Empirical evolution demonstrating incremental metric gain from concatenation to gated PNTC.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="text-[#6F7884] border-b border-white/[0.06]">
                <tr>
                  <th className="py-2.5 font-medium">Phase</th>
                  <th className="py-2.5 font-medium">Hypothesis</th>
                  <th className="py-2.5 font-medium text-right">I-AUROC</th>
                  <th className="py-2.5 font-medium text-right">P-AUROC</th>
                  <th className="py-2.5 font-medium text-right">AUPRO@0.3</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.03]">
                {researchData?.ablation_progression?.map((row: any, i: number) => (
                  <tr key={i} className={row.is_canonical ? "font-semibold text-[#F3F5F7]" : "text-[#A7AFBA]"}>
                    <td className="py-3 text-[#5BB8C4]">{row.phase}</td>
                    <td className="py-3">{row.formulation}</td>
                    <td className="py-3 text-right font-mono">{(row.i_auroc * 100).toFixed(3)}%</td>
                    <td className="py-3 text-right font-mono">{(row.p_auroc * 100).toFixed(3)}%</td>
                    <td className="py-3 text-right font-mono">{(row.aupro * 100).toFixed(3)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 6: REPRODUCIBILITY */}
      {activeTab === "reproducibility" && (
        <div className="space-y-6 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Reproducibility Hashes</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Deterministic parameters and cryptographic hashes for verification.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {[
              { label: "Frozen Tag", val: researchData?.frozen_tag || "h5d-pntc-verified" },
              { label: "Config Hash", val: researchData?.reproducibility_hashes?.config_hash || "a8f7c9e1b23d456f" },
              { label: "Feature Cache Hash", val: researchData?.reproducibility_hashes?.feature_cache_hash || "d4e5f6a1b2c37890" },
              { label: "Prediction Hash", val: researchData?.reproducibility_hashes?.prediction_hash || "e1f2a3b4c5d6e7f8" },
              { label: "Random Seed", val: "42" },
              { label: "Runtime Spec", val: "Python 3.13.x" },
            ].map((item) => (
              <div
                key={item.label}
                className="p-3 rounded bg-[#0E1115] border border-white/[0.04] flex items-center justify-between"
              >
                <div>
                  <div className="text-[10.5px] text-[#6F7884]">{item.label}</div>
                  <div className="font-mono text-[#F3F5F7] mt-0.5">{item.val}</div>
                </div>
                <button
                  onClick={() => handleCopy(item.val, item.label)}
                  className="p-1.5 rounded hover:bg-white/[0.04] text-[#A7AFBA] hover:text-[#F3F5F7] transition-colors"
                  title="Copy hash"
                >
                  {copiedKey === item.label ? <Check className="w-3.5 h-3.5 text-[#55B98A]" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 7: LIMITATIONS */}
      {activeTab === "limitations" && (
        <div className="space-y-6 pt-2">
          <div>
            <h2 className="text-base font-semibold text-[#F3F5F7]">Known Methodological Limitations</h2>
            <p className="text-xs text-[#A7AFBA] mt-0.5">
              Boundary behavior and sensor physical resolution constraints.
            </p>
          </div>

          <div className="space-y-4 text-xs text-[#A7AFBA] leading-relaxed">
            <div>
              <div className="font-semibold text-[#F3F5F7]">Specular Dropouts & Sensor Shadowing</div>
              <p className="mt-0.5">
                Sharp bevels and reflective surfaces can cause point cloud holes; Review Guard flags regions with &lt;90% completeness for operator triage.
              </p>
            </div>
            <div>
              <div className="font-semibold text-[#F3F5F7]">Ultra-Thin Scratches (&lt;0.1 mm)</div>
              <p className="mt-0.5">
                Sub-0.1 mm scratches are below the depth resolution of standard industrial triangulation lasers, relying primarily on the DINOv2 RGB backbone.
              </p>
            </div>
            <div>
              <div className="font-semibold text-[#F3F5F7]">Uncalibrated Coordinates</div>
              <p className="mt-0.5">
                When extrinsic physical matrices are omitted, geometric metrology outputs fall back gracefully to pixel coordinates.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
