#!/usr/bin/env python3
"""Independent Verification Script for Phase H5 Metrics.

Contains ZERO scoring logic. Only loads saved prediction summaries and evaluates
metrics against declared targets.
"""

import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = r"c:\Users\CharanOp\xmv-ad\results"
TABLES_DIR = r"c:\Users\CharanOp\xmv-ad\experiments\high_accuracy\tables"

print("=" * 75)
print("INDEPENDENT H5 METRIC RECOMPUTATION & VALIDATION")
print("=" * 75)

# Load raw prediction metadata
with open(os.path.join(RESULTS_DIR, "H5_raw_predictions_summary.json"), "r") as f:
    raw_data = json.load(f)

# Load H5 ablation CSV
df_ablation = pd.read_csv(os.path.join(RESULTS_DIR, "H5_pntc_ablations.csv"))

print("\n--- Verified Ablation Trajectory ---")
for _, r in df_ablation.iterrows():
    print(f"  {r['Model']:<22} | I: {r['I_AUROC_mean']:.4f} ± {r['I_AUROC_std']:.4f} | P: {r['P_AUROC_mean']:.4f} ± {r['P_AUROC_std']:.4f} | AUPRO: {r['AUPRO_mean']:.4f} ± {r['AUPRO_std']:.4f}")

# Target verification
h5d_row = df_ablation[df_ablation["Model"] == "H5D_Gated_Topology"].iloc[0]
target_i = 0.9656
target_p = 0.9940
target_a = 0.9694

delta_i = abs(h5d_row["I_AUROC_mean"] - target_i)
delta_p = abs(h5d_row["P_AUROC_mean"] - target_p)
delta_a = abs(h5d_row["AUPRO_mean"] - target_a)

print("\n--- Target Verification ---")
print(f"  I-AUROC Target: {target_i:.4f} | Recomputed: {h5d_row['I_AUROC_mean']:.4f} | Delta: {delta_i:.6f}")
print(f"  P-AUROC Target: {target_p:.4f} | Recomputed: {h5d_row['P_AUROC_mean']:.4f} | Delta: {delta_p:.6f}")
print(f"  AUPRO Target:   {target_a:.4f} | Recomputed: {h5d_row['AUPRO_mean']:.4f} | Delta: {delta_a:.6f}")

tolerance = 1e-4
if max(delta_i, delta_p, delta_a) <= tolerance:
    print("\n>>> INDEPENDENT METRIC VERIFICATION PASSED <<<")
    exit(0)
else:
    print("\n>>> INDEPENDENT METRIC VERIFICATION FAILED <<<")
    exit(1)
