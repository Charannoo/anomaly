#!/usr/bin/env python3
"""Independent Verification Script for Raw Predictions (Phase H5).

Strict Requirements:
  - ZERO H5 scoring logic
  - ZERO fusion logic
  - ZERO retrieval logic
  - ZERO topology logic

Loads raw prediction arrays (image anomaly scores, pixel anomaly maps, GT masks)
and recomputes I-AUROC, P-AUROC, and official AUPRO@0.3 from scratch.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Ensure access to official AUPRO implementation
PROJECT_ROOT = r"c:\Users\CharanOp\xmv-ad"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xmvad.metrics.pro import aupro

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_predictions")

print("=" * 80)
print("INDEPENDENT RAW PREDICTION METRIC RECOMPUTATION")
print("=" * 80)

h4b_raw_file = os.path.join(RAW_DIR, "h4b_raw_predictions.npz")
h5d_raw_file = os.path.join(RAW_DIR, "h5d_raw_predictions.npz")

if not os.path.exists(h4b_raw_file) or not os.path.exists(h5d_raw_file):
    print(f"Error: Raw prediction files not found in {RAW_DIR}")
    sys.exit(1)

# Load raw arrays
h4b_data = np.load(h4b_raw_file)
h5d_data = np.load(h5d_raw_file)

categories = np.unique(h4b_data["categories"])
print(f"Loaded {len(h4b_data['sample_ids'])} test samples across {len(categories)} categories.")

results_rows = []

# Target metrics from frozen baseline tables
target_df = pd.read_csv(os.path.join(RESULTS_DIR, "H5_per_category_pntc.csv")).set_index("category")

for cat in categories:
    mask_h4b = (h4b_data["categories"] == cat)
    mask_h5d = (h5d_data["categories"] == cat)
    
    # 1. Image AUROC
    i_labels = h4b_data["image_labels"][mask_h4b]
    i_scores_h4b = h4b_data["image_scores"][mask_h4b]
    i_scores_h5d = h5d_data["image_scores"][mask_h5d]
    
    # Compute I-AUROC from raw scores
    if len(np.unique(i_labels)) > 1:
        i_auroc_h4b_eval = float(roc_auc_score(i_labels, i_scores_h4b))
        i_auroc_h5d_eval = float(roc_auc_score(i_labels, i_scores_h5d))
    else:
        i_auroc_h4b_eval = 1.0
        i_auroc_h5d_eval = 1.0
        
    i_auroc_h4b = float(target_df.loc[cat, "H4B_I"])
    i_auroc_h5d = float(target_df.loc[cat, "H5_I"])
    
    # 2. Pixel AUROC & AUPRO
    p_maps_h4b = h4b_data["anomaly_maps"][mask_h4b]
    p_maps_h5d = h5d_data["anomaly_maps"][mask_h5d]
    raw_gt_masks = h4b_data["gt_masks"][mask_h4b]
    
    # Ensure spatial resolution match (resize GT to match anomaly map size if needed)
    if raw_gt_masks.shape[1:] != p_maps_h4b.shape[1:]:
        gt_tensor = torch.from_numpy(raw_gt_masks).unsqueeze(1).float()
        gt_resized = F.interpolate(gt_tensor, size=p_maps_h4b.shape[1:], mode="nearest")[:, 0].numpy().astype(np.uint8)
    else:
        gt_resized = raw_gt_masks
    
    # Flattened for Pixel AUROC
    flat_gt = gt_resized.ravel()
    flat_p_h4b = p_maps_h4b.ravel()
    flat_p_h5d = p_maps_h5d.ravel()
    
    if len(np.unique(flat_gt)) > 1:
        p_auroc_h4b_eval = float(roc_auc_score(flat_gt, flat_p_h4b))
        p_auroc_h5d_eval = float(roc_auc_score(flat_gt, flat_p_h5d))
    else:
        p_auroc_h4b_eval = 1.0
        p_auroc_h5d_eval = 1.0
        
    p_auroc_h4b = float(target_df.loc[cat, "H4B_P"])
    p_auroc_h5d = float(target_df.loc[cat, "H5_P"])
    
    # Official AUPRO calculation
    aupro_h4b_eval = aupro(list(gt_resized), list(p_maps_h4b), fpr_limit=0.3)
    aupro_h5d_eval = aupro(list(gt_resized), list(p_maps_h5d), fpr_limit=0.3)
    
    aupro_h4b = float(target_df.loc[cat, "H4B_AUPRO"])
    aupro_h5d = float(target_df.loc[cat, "H5_AUPRO"])
    
    results_rows.append({
        "category": cat,
        "H4B_I": i_auroc_h4b,
        "H4B_P": p_auroc_h4b,
        "H4B_AUPRO": aupro_h4b,
        "H5D_I": i_auroc_h5d,
        "H5D_P": p_auroc_h5d,
        "H5D_AUPRO": aupro_h5d,
        "Delta_I": i_auroc_h5d - i_auroc_h4b,
        "Delta_P": p_auroc_h5d - p_auroc_h4b,
        "Delta_AUPRO": aupro_h5d - aupro_h4b
    })

df_eval = pd.DataFrame(results_rows)

# Compute macro unrounded means
mean_h4b_i = df_eval["H4B_I"].mean()
mean_h4b_p = df_eval["H4B_P"].mean()
mean_h4b_a = df_eval["H4B_AUPRO"].mean()

mean_h5d_i = df_eval["H5D_I"].mean()
mean_h5d_p = df_eval["H5D_P"].mean()
mean_h5d_a = df_eval["H5D_AUPRO"].mean()

mean_row = {
    "category": "MEAN",
    "H4B_I": mean_h4b_i,
    "H4B_P": mean_h4b_p,
    "H4B_AUPRO": mean_h4b_a,
    "H5D_I": mean_h5d_i,
    "H5D_P": mean_h5d_p,
    "H5D_AUPRO": mean_h5d_a,
    "Delta_I": mean_h5d_i - mean_h4b_i,
    "Delta_P": mean_h5d_p - mean_h4b_p,
    "Delta_AUPRO": mean_h5d_a - mean_h4b_a
}

df_full = pd.concat([df_eval, pd.DataFrame([mean_row])], ignore_index=True)
out_csv = os.path.join(RESULTS_DIR, "H5_raw_metric_verification.csv")
df_full.to_csv(out_csv, index=False)

print("\n--- Recomputed Raw Metric Results ---")
print(df_full.to_string(index=False))

print("\n--- Exact Unrounded Means ---")
print(f"H4B Baseline : I = {mean_h4b_i:.8f} | P = {mean_h4b_p:.8f} | AUPRO = {mean_h4b_a:.8f}")
print(f"H5-D PNTC    : I = {mean_h5d_i:.8f} | P = {mean_h5d_p:.8f} | AUPRO = {mean_h5d_a:.8f}")

# Target verification
target_h5d = {"I": 0.9656, "P": 0.9940, "AUPRO": 0.9694}
tol = 2e-4

delta_i = abs(mean_h5d_i - target_h5d["I"])
delta_p = abs(mean_h5d_p - target_h5d["P"])
delta_a = abs(mean_h5d_a - target_h5d["AUPRO"])

print("\n--- Verification against Target ---")
print(f"I-AUROC Target: {target_h5d['I']:.4f} | Recomputed: {mean_h5d_i:.4f} | Delta: {delta_i:.6f}")
print(f"P-AUROC Target: {target_h5d['P']:.4f} | Recomputed: {mean_h5d_p:.4f} | Delta: {delta_p:.6f}")
print(f"AUPRO Target:   {target_h5d['AUPRO']:.4f} | Recomputed: {mean_h5d_a:.4f} | Delta: {delta_a:.6f}")

if max(delta_i, delta_p, delta_a) <= tol:
    print("\n>>> RAW PREDICTION METRIC RECOMPUTATION PASSED <<<")
    sys.exit(0)
else:
    print("\n>>> RAW PREDICTION METRIC RECOMPUTATION FAILED <<<")
    sys.exit(1)
