"""H0.9 - Build H0 M3DM reproduction comparison table.

Columns: per-category paper-reported M3DM metrics (RGB+3D full setting, from the
primary source arXiv:2303.00601v2) vs our measured metrics and per-category delta.

Our measured values are intentionally "NA (not runnable)" on this CPU-only
machine; we never fabricate a number. Means are computed from the paper columns.

Outputs:
  experiments/high_accuracy/tables/H0_m3dm_reproduction.csv
  experiments/high_accuracy/tables/H0_m3dm_paper_metrics.md
"""
import csv
import os

CATS = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]

# Primary source: arXiv:2303.00601v2 (CVPR 2023), M3DM "Ours" RGB+3D setting.
# I-AUROC: main paper Table 1 (RGB+3D). AUPRO: main paper Table 2 (RGB+3D).
# P-AUROC: main paper Appendix B, Table I (RGB+3D, "Ours").
PAPER = {
    "bagel":        {"i_auroc": 0.994, "p_auroc": 0.995, "aupro": 0.970},
    "cable_gland":  {"i_auroc": 0.909, "p_auroc": 0.993, "aupro": 0.971},
    "carrot":       {"i_auroc": 0.972, "p_auroc": 0.997, "aupro": 0.979},
    "cookie":       {"i_auroc": 0.976, "p_auroc": 0.985, "aupro": 0.950},
    "dowel":        {"i_auroc": 0.960, "p_auroc": 0.985, "aupro": 0.941},
    "foam":         {"i_auroc": 0.942, "p_auroc": 0.984, "aupro": 0.932},
    "peach":        {"i_auroc": 0.973, "p_auroc": 0.996, "aupro": 0.977},
    "potato":       {"i_auroc": 0.899, "p_auroc": 0.994, "aupro": 0.971},
    "rope":         {"i_auroc": 0.972, "p_auroc": 0.997, "aupro": 0.971},
    "tire":         {"i_auroc": 0.850, "p_auroc": 0.996, "aupro": 0.975},
}

OUT_DIR = os.path.join("experiments", "high_accuracy", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

ROWS = []
for cat in CATS:
    p = PAPER[cat]
    ROWS.append({
        "category": cat,
        "paper_I_AUROC": p["i_auroc"],
        "paper_P_AUROC": p["p_auroc"],
        "paper_AUPRO": p["aupro"],
        "measured_I_AUROC": "NA",
        "measured_P_AUROC": "NA",
        "measured_AUPRO": "NA",
        "delta_I_AUROC": "NA",
        "delta_P_AUROC": "NA",
        "delta_AUPRO": "NA",
        "status": "measurement blocked: no CUDA GPU",
    })

mean_i = round(sum(PAPER[c]["i_auroc"] for c in CATS) / len(CATS), 3)
mean_p = round(sum(PAPER[c]["p_auroc"] for c in CATS) / len(CATS), 3)
mean_a = round(sum(PAPER[c]["aupro"] for c in CATS) / len(CATS), 3)
ROWS.append({
    "category": "Mean",
    "paper_I_AUROC": mean_i,
    "paper_P_AUROC": mean_p,
    "paper_AUPRO": mean_a,
    "measured_I_AUROC": "NA",
    "measured_P_AUROC": "NA",
    "measured_AUPRO": "NA",
    "delta_I_AUROC": "NA",
    "delta_P_AUROC": "NA",
    "delta_AUPRO": "NA",
    "status": "measurement blocked",
})

csv_path = os.path.join(OUT_DIR, "H0_m3dm_reproduction.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    writer.writeheader()
    writer.writerows(ROWS)

md_path = os.path.join(OUT_DIR, "H0_m3dm_paper_metrics.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# H0: M3DM metrics on MVTec-3D AD (REPORTED BY PRIOR WORK)\n\n")
    f.write("Source: **arXiv:2303.00601v2** (Multimodal Industrial Anomaly Detection\n")
    f.write("via Hybrid Fusion, CVPR 2023), M3DM full setting (DINO+Point_MAE+Fusion,\n")
    f.write("RGB+3D, 3 memory banks + UFF + DLF). Primary-source tables only:\n")
    f.write("I-AUROC = main Table 1; AUPRO = main Table 2; P-AUROC = App. B Table I.\n\n")
    f.write("| Category | I-AUROC | P-AUROC | AUPRO |\n")
    f.write("|---|---:|---:|---:|\n")
    for r in ROWS:
        f.write(f"| {r['category']} | {r['paper_I_AUROC']:.3f} | "
                f"{r['paper_P_AUROC']:.3f} | {r['paper_AUPRO']:.3f} |\n")

print("wrote", csv_path)
print("wrote", md_path)
print(f"computed paper means: I-AUROC={mean_i:.3f} P-AUROC={mean_p:.3f} AUPRO={mean_a:.3f}")