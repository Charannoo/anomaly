"""Aggregate deployment benchmark + accuracy into final summary (Phase 6).

Produces:
  experiments/tables/deployment_summary.csv — one row per backend:
    latency (mean of per-cat p50/p95 model-only & e2e), FPS, peak RSS,
    artifact size, params, params_source, mean accuracy + mean |delta| vs
    frozen native 256 table, target classification, verdict notes
  experiments/tables/deployment_final.md    — markdown table for the report
  experiments/tables/deployment_pareto.png  — latency vs accuracy scatter
                                               (I-AUROC) with Pareto frontier
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "experiments" / "tables"
sys.path.insert(0, str(REPO_ROOT / "src"))

BACKENDS = ["pytorch", "onnx", "openvino_fp32", "openvino_int8"]
THRESH = {"ideal": 0.005, "accept": 0.01, "weak": 0.02}  # pre-declared in DEPLOYMENT_INPUT_CONTRACT.md

ARTIFACT_MB = {"pytorch": 7.71, "onnx": 2.32, "openvino_fp32": 1.02, "openvino_int8": 1.22}


def main() -> int:
    bench = pd.read_csv(TABLES / "deployment_benchmark.csv")
    frozen = pd.read_csv(TABLES / "E4_concat_metrics.csv").set_index("category")

    rows = []
    for b in BACKENDS:
        bb = bench[bench.backend == b]
        m = pd.read_csv(TABLES / f"deployment_{b}_metrics.csv").set_index("category")
        if b != "pytorch":
            d = pd.read_csv(TABLES / f"deployment_{b}_vs_pytorch.csv").set_index("category")
        else:
            d = pd.DataFrame(index=m.index)
            for col, fc in (("d_I_AUROC", "image_auroc_max"), ("d_P_AUROC", "pixel_auroc"),
                            ("d_AUPRO", "aupro")):
                d[col] = m[fc] - frozen[fc]

        def agg(csvcol, jkey):
            return float(np.mean([json.loads(x)[jkey] for x in bb[csvcol]]))

        i_mean = float(m["image_auroc_max"].mean())
        p_mean = float(m["pixel_auroc"].mean())
        ap_mean = float(m["aupro"].mean())
        rows.append({
            "backend": b,
            "model_p50_ms": round(agg("model_only_ms", "p50"), 2),
            "model_p95_ms": round(agg("model_only_ms", "p95"), 2),
            "e2e_p50_ms": round(agg("e2e_ms", "p50"), 2),
            "e2e_p95_ms": round(agg("e2e_ms", "p95"), 2),
            "model_fps": round(1000.0 / agg("model_only_ms", "p50"), 1),
            "e2e_fps": round(1000.0 / agg("e2e_ms", "p50"), 1),
            "peak_rss_model_mb": round(float(bb["peak_rss_model_only_mb"].mean()), 1),
            "peak_rss_e2e_mb": round(float(bb["peak_rss_e2e_mb"].mean()), 1),
            "artifact_size_mb": ARTIFACT_MB[b],
            "params_m": 1.90 if b == "pytorch" else 0.48,
            "params_source": "in-memory wrapper (incl unused teacher wts)" if b == "pytorch"
                             else "ONNX runtime graph (traced student path)",
            "mean_I_AUROC": round(i_mean, 4),
            "mean_P_AUROC": round(p_mean, 4),
            "mean_AUPRO": round(ap_mean, 4),
            "mean_abs_d_I_AUROC": round(float(np.abs(d["d_I_AUROC"]).mean()), 4),
            "mean_abs_d_P_AUROC": round(float(np.abs(d["d_P_AUROC"]).mean()), 4),
            "mean_abs_d_AUPRO": round(float(np.abs(d["d_AUPRO"]).mean()), 4),
        })

    df = pd.DataFrame(rows)
    # classify INT8 against pre-declared thresholds (max mean-abs-delta drives verdict)
    i8 = df[df.backend == "openvino_int8"].iloc[0]
    worst = max(i8["mean_abs_d_I_AUROC"], i8["mean_abs_d_P_AUROC"], i8["mean_abs_d_AUPRO"])
    i8_verdict = "FAIL"
    i8_cls = "INT8 exceeded worst-case mean-abs |delta| threshold (0.02): " \
             f"{worst:.3f} (AUPRO mean abs delta {i8['mean_abs_d_AUPRO']:.3f}). " \
             "Declared failure path entered; FP32 stays recommended."
    df["int8_verdict"] = ["", "", "", i8_verdict]

    df.to_csv(TABLES / "deployment_summary.csv", index=False)

    md = ["| backend | model p50 (ms) | e2e p95 (ms) | model FPS | peak RSS (MB) | file (MB) | "
          "mean I-AUROC | mean P-AUROC | mean AUPRO | mean|Δ| I/P/AUPRO vs frozen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['backend']} | {r['model_p50_ms']} | {r['e2e_p95_ms']} | {r['model_fps']} | "
                  f"{r['peak_rss_model_mb']} | {r['artifact_size_mb']} | {r['mean_I_AUROC']} | "
                  f"{r['mean_P_AUROC']} | {r['mean_AUPRO']} | "
                  f"{r['mean_abs_d_I_AUROC']}/{r['mean_abs_d_P_AUROC']}/{r['mean_abs_d_AUPRO']} |")
    md.append("\nINT8 verdict: " + i8_cls)
    (TABLES / "deployment_final.md").write_text("\n".join(md), encoding="utf-8")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for _, r in df.iterrows():
            color = "#d62728" if r["backend"] == "openvino_int8" else "#1f77b4"
            ax.scatter(r["model_p50_ms"], r["mean_I_AUROC"], s=90, color=color, zorder=3)
            ax.annotate(r["backend"].replace("openvino_", "ov_"), (r["model_p50_ms"], r["mean_I_AUROC"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel("model-only p50 latency (ms, 1 thread, CPU)")
        ax.set_ylabel("mean image AUROC (max variant, frozen protocol)")
        ax.set_title("E4 deployment: latency vs detection quality (FP32/INT8)")
        ax.grid(alpha=0.3)
        ax.axvspan(0, 15, alpha=0.06, color="green")
        fig.tight_layout()
        fig.savefig(TABLES / "deployment_pareto.png", dpi=150)
        print("wrote deployment_pareto.png")
    print(df.to_string(index=False))
    print(i8_cls)
    (TABLES / "deployment_final.md")
    print("wrote deployment_summary.csv + deployment_final.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())