"""Phase 5, todo 5: build the paper-oriented final baseline table.

Rows: MVTec-3D-AD-2022 base-paper baselines (provenance "Reported by prior
paper", transcribed in docs/BASE_PAPER_2022.md), M3DM 2023 (later comparator,
labeled not-directly-comparable), and the XMV-AD pipeline E1/E2/E3A/E4/E7
(provenance "Measured by us").

Columns: Method, Group, Input representation, I-AUROC, P-AUROC, AUPRO, plus
XMV-AD-only columns (trainable params, checkpoint size, CPU latency batch1@256,
modality-explanation support), provenance, and notes.

Writes experiments/tables/paper_baseline_table.csv (machine-readable) and
docs/TABLES_PAPER_BASELINE.md (human-readable, paper-formatted).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reported-by-prior-paper numbers (transcribed in docs/BASE_PAPER_2022.md)
# ---------------------------------------------------------------------------
# (method, input, I-AUROC, P-AUROC, AUPRO); "--" = not reported in paper.
BASE_PAPER_ROWS = [
    ("Voxel f-AnoGAN", "3D only", 0.538, "--", 0.583),
    ("Voxel f-AnoGAN", "3D+RGB", 0.517, "--", 0.639),
    ("Voxel AE", "3D+RGB", 0.538, "--", 0.564),
    ("Voxel Variation Model", "3D only", 0.699, "--", 0.492),
    ("Voxel Variation Model", "3D+RGB", 0.609, "--", 0.471),
    ("Depth Variation Model", "3D+RGB", 0.555, "--", 0.335),
]
M3DM_ROWS = [
    ("M3DM (Wang et al., CVPR 2023)", "RGB+3D point cloud", 0.945, "--", 0.964),
]


def xmv_rows() -> list[dict]:
    """E1/E2/E3A/E4/E7 metrics from frozen project tables."""
    tables = REPO_ROOT / "experiments" / "tables"
    e1 = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    e2 = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    e3 = pd.read_csv(tables / "E3_vs_baselines.csv")
    e7 = pd.read_csv(tables / "E7_xmv_bidirectional_metrics.csv")
    e4n = pd.read_csv(tables / "E4_concat_metrics_native.csv")

    def mean_of(df, key, exclude="MEAN"):
        d = df[df.category != exclude]
        return float(np.nanmean(d[key].to_numpy(float)))

    def fixed_mean(metric):
        d = e3[(e3.category != "MEAN") & (e3.metric == metric)]
        return float(np.nanmean(d["LateFusion"].to_numpy(float)))

    e1_score = lambda k: mean_of(e1, k)  # noqa: E731
    e2_score = lambda k: mean_of(e2, k)  # noqa: E731
    e4_mean = e4n[e4n.res == "native"]
    # E4 at native resolution (official full-res comparison) and at 256.
    e4 = {
        "i256": mean_of(pd.read_csv(tables / "E4_concat_metrics.csv"), "image_auroc_max"),
        "p256": mean_of(pd.read_csv(tables / "E4_concat_metrics.csv"), "pixel_auroc"),
        "a256": mean_of(pd.read_csv(tables / "E4_concat_metrics.csv"), "aupro"),
        "in": mean_of(e4_mean, "image_auroc_max"),
        "pn": mean_of(e4_mean, "pixel_auroc"),
        "an": mean_of(e4_mean, "aupro"),
    }
    e7i = float(e7[e7.category == "MEAN"]["image_auroc_max"].iloc[0])
    e7p = float(e7[e7.category == "MEAN"]["pixel_auroc"].iloc[0])
    e7a = float(e7[e7.category == "MEAN"]["aupro"].iloc[0])

    e1_sz = _avg_ckpt_kb("checkpoints/E1_baseline_rgb")
    e2_sz = _avg_ckpt_kb("checkpoints/E2_baseline_depth")

    rows = [
        {"method": "E1 RGB (Phase 2 student)", "group": "XMV-AD pipeline",
         "input": "RGB only", "i_auroc": round(e1_score("image_auroc_max"), 4),
         "p_auroc": round(e1_score("pixel_auroc"), 4), "aupro": round(e1_score("aupro"), 4),
         "params": 959097, "ckpt_kb": round(e1_sz, 1), "latency_ms": 14.8,
         "explanation": "No (single modality)",
         "resolution": "256", "provenance": "Measured by us"},
        {"method": "E2 Depth (Phase 2 student)", "group": "XMV-AD pipeline",
         "input": "Z-depth only", "i_auroc": round(e2_score("image_auroc_max"), 4),
         "p_auroc": round(e2_score("pixel_auroc"), 4), "aupro": round(e2_score("aupro"), 4),
         "params": 959097, "ckpt_kb": round(e2_sz, 1), "latency_ms": 15.0,
         "explanation": "No (single modality)",
         "resolution": "256", "provenance": "Measured by us"},
        {"method": "E3A Fixed fusion (ex-pred score mean)", "group": "XMV-AD pipeline",
         "input": "RGB+Z-depth (late)", "i_auroc": round(fixed_mean("I-AUROC(max)"), 4),
         "p_auroc": round(fixed_mean("P-AUROC"), 4), "aupro": round(fixed_mean("AUPRO"), 4),
         "params": 0, "ckpt_kb": 0.0, "latency_ms": 30.3,
         "explanation": "Partial (heatmap only)",
         "resolution": "256", "provenance": "Measured by us"},
        {"method": "E4 XMV-AD Residual Fusion", "group": "XMV-AD pipeline",
         "input": "RGB+Z-depth (feature)", "i_auroc": round(e4["in"], 4),
         "p_auroc": round(e4["pn"], 4), "aupro": round(e4["an"], 4),
         "params": 8738, "ckpt_kb": 42.3, "latency_ms": 34.0,
         "explanation": "Yes (A_RGB / A_DEPTH / A_FINAL + relative evidence)",
         "resolution": "native-GT", "provenance": "Measured by us"},
        {"method": "E7 XMV-Attention Ablation", "group": "XMV-AD ablation",
         "input": "RGB+Z-depth (feature)", "i_auroc": round(e7i, 4),
         "p_auroc": round(e7p, 4), "aupro": round(e7a, 4),
         "params": 12770, "ckpt_kb": 60.2, "latency_ms": 38.0,
         "explanation": "Yes (evidence) | ablated cross-attention",
         "resolution": "256", "provenance": "Measured by us"},
    ]
    return rows


def _avg_ckpt_kb(directory: str) -> float:
    import os

    files = list((REPO_ROOT / directory).glob("*_best.pt"))
    if not files:
        return 0.0
    return float(np.mean([os.path.getsize(f) for f in files])) / 1024.0


def main() -> int:
    rows: list[dict] = []
    for method, inp, i, p, a in BASE_PAPER_ROWS:
        rows.append({"method": method, "group": "MVTec-3D-AD 2022 (initial benchmark)",
                     "input": inp, "i_auroc": i, "p_auroc": p, "aupro": a,
                     "params": "--", "ckpt_kb": "--", "latency_ms": "--",
                     "explanation": "Not applicable", "resolution": "native",
                     "provenance": "Reported by prior paper"})
    for method, inp, i, p, a in M3DM_ROWS:
        rows.append({"method": method, "group": "Later comparator (not directly comparable)",
                     "input": inp, "i_auroc": i, "p_auroc": p, "aupro": a,
                     "params": "--", "ckpt_kb": "--", "latency_ms": "--",
                     "explanation": "N/A (pre-trained ViT + point-cloud memory banks)",
                     "resolution": "native", "provenance": "Reported by prior paper"})
    rows.extend(xmv_rows())

    df = pd.DataFrame(rows, columns=[
        "method", "group", "input", "i_auroc", "p_auroc", "aupro",
        "params", "ckpt_kb", "latency_ms", "explanation", "resolution",
        "provenance"])
    tables = REPO_ROOT / "experiments" / "tables"
    df.to_csv(tables / "paper_baseline_table.csv", index=False)

    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    lines = [
        "# Paper baseline table (Phase 5)",
        "",
        "All values carry a provenance label. '--' means the value is not",
        "reported in the cited source. AUPRO uses FPR limit 0.3 everywhere;",
        "RESOLUTION notes at which map resolution the pixel metrics were",
        "evaluated (official full-resolution vs 256).",
        "",
        "| Method | Input | I-AUROC | P-AUROC | AUPRO | Params | Size (KB) | CPU lat. (ms) | Modality explanation | Res | Provenance |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r.method} | {r.input} | {r.i_auroc} | {r.p_auroc} | {r.aupro} "
                     f"| {r.params} | {r.ckpt_kb} | {r.latency_ms} | {r.explanation} "
                     f"| {r.resolution} | {r.provenance} |")
    (docs / "TABLES_PAPER_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {tables / 'paper_baseline_table.csv'} and {docs / 'TABLES_PAPER_BASELINE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())