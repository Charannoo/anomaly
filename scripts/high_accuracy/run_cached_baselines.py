#!/usr/bin/env python3
"""Cached Baseline Runner for XMV-AD-H.

Runs H1 (RGB DINO), H2 (Point-MAE XYZ), and H3 (Multimodal Baselines)
PURELY from canonical cached features without rerunning backbone neural networks.
"""

import os
import sys
import time
import glob
import math
import hashlib
import argparse
import csv
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import random_projection
import scipy.sparse as sp

# Ensure M3DM modules can be imported
sys.path.insert(0, '/opt/m3dm_ws/m3dm')
if os.path.exists('/opt/m3dm_ws/m3dm'):
    os.chdir('/opt/m3dm_ws/m3dm')

from utils.utils import KNNGaussianBlur
from utils.au_pro_util import calculate_au_pro

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire"
]

CORESET_CACHE_DIR = "/opt/tmp/coreset_cache"
os.makedirs(CORESET_CACHE_DIR, exist_ok=True)

def get_coreset_indices(z_lib, f_coreset=0.1, coreset_eps=0.9, random_state=42, cache_tag=""):
    N, D = z_lib.shape
    n = max(1, int(f_coreset * N))
    
    # Check cache
    cache_path = os.path.join(CORESET_CACHE_DIR, f"coreset_{cache_tag}_{N}_{D}_{n}.pt")
    if os.path.exists(cache_path):
        print(f"   [CORESET CACHE HIT] {cache_tag} ({N}, {D}) -> {n} indices")
        return torch.load(cache_path, map_location="cpu")
        
    print(f"   [CORESET FIT] {cache_tag} ({N}, {D}) -> selecting {n} items...")
    transformer = random_projection.SparseRandomProjection(eps=coreset_eps, random_state=random_state)
    transformer.fit(sp.csr_matrix((N, D), dtype=np.float64))
    
    CHUNK = 50_000
    parts = []
    with torch.inference_mode():
        for start in range(0, N, CHUNK):
            seg = z_lib[start:start + CHUNK].detach().cpu().numpy().astype(np.float32)
            t = transformer.transform(seg)
            t = torch.from_numpy(np.asarray(t, dtype=np.float32)).half()
            parts.append(t)
    proj_lib = torch.cat(parts, dim=0)
    del parts
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proj_lib = proj_lib.to(dev)
    
    select_idx = 0
    last_item = proj_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    
    Nc = proj_lib.shape[0]
    min_distances = torch.full((Nc, 1), float("inf"), dtype=torch.float16, device=dev)
    B = 100_000
    
    def _chunked_norm(t, ref, out):
        with torch.inference_mode():
            for s in range(0, Nc, B):
                d = torch.linalg.norm(t[s:s + B] - ref, dim=1, keepdims=True)
                out[s:s + B] = d
        return out
        
    min_distances = _chunked_norm(proj_lib, last_item, min_distances)
    
    for _ in range(n - 1):
        distances = torch.linalg.norm(proj_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)
        last_item = proj_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))
        
    result = torch.stack(coreset_idx)
    torch.save(result, cache_path)
    return result

def compute_patch_anomaly_map(test_patch, mem_bank, feat_dim_grid=(28, 28), n_reweight=3, blur_filter=None):
    """Computes sample anomaly score and 224x224 anomaly segmentation map."""
    # test_patch: (N_patches, D), mem_bank: (N_bank, D)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_patch_d = test_patch.to(dev)
    mem_bank_d = mem_bank.to(dev)
    
    with torch.inference_mode():
        # Chunked cdist to bound VRAM
        dist = torch.cdist(test_patch_d, mem_bank_d) # (N_patches, N_bank)
        min_val, min_idx = torch.min(dist, dim=1) # (N_patches,)
        
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        
        # Reweighting
        m_test = test_patch_d[s_idx].unsqueeze(0)
        m_star = mem_bank_d[min_idx[s_idx]].unsqueeze(0)
        w_dist = torch.cdist(m_star, mem_bank_d)
        _, nn_idx = torch.topk(w_dist, k=n_reweight, largest=False)
        
        m_star_knn = torch.linalg.norm(m_test - mem_bank_d[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(test_patch.shape[1], dtype=torch.float32, device=dev))
        w = 1.0 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = (w * s_star).cpu().item()
        
        # Segmentation map
        H, W = feat_dim_grid
        s_map = min_val.view(1, 1, H, W)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear', align_corners=False)
        if blur_filter is not None:
            s_map = blur_filter(s_map)
        s_map_np = s_map.squeeze().cpu().numpy()
        
    return s, s_map_np

class CachedEvaluator:
    def __init__(self, cache_root="/opt/xmv_h_fast/canonical"):
        self.cache_root = cache_root
        self.blur = KNNGaussianBlur(4)
        print("BACKBONE EXTRACTION: SKIPPED (CACHE HIT)")
        print(f"Loading cached descriptors from: {cache_root}")

    def load_category_samples(self, category, split="train"):
        split_dir = os.path.join(self.cache_root, category, split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        samples = []
        for f in files:
            s = torch.load(f, map_location="cpu")
            samples.append(s)
        return samples

    def evaluate_h1_rgb(self, category):
        """H1: DINO ViT-B/8 RGB Memory Bank Baseline."""
        train_samples = self.load_category_samples(category, "train")
        test_samples = self.load_category_samples(category, "test")
        
        # Build Train Bank
        rgb_patches = [s["rgb_patch"] for s in train_samples] # each (784, 768)
        patch_lib = torch.cat(rgb_patches, dim=0) # (N_train*784, 768)
        
        # Normalize
        mean = torch.mean(patch_lib)
        std = torch.std(patch_lib)
        patch_lib = (patch_lib - mean) / (std + 1e-7)
        
        # Coreset
        coreset_idx = get_coreset_indices(patch_lib, f_coreset=0.1, cache_tag=f"{category}_rgb")
        coreset_bank = patch_lib[coreset_idx]
        
        image_preds, image_labels = [], []
        pixel_preds, pixel_labels = [], []
        gts, predictions = [], []
        
        for s in test_samples:
            p = (s["rgb_patch"] - mean) / (std + 1e-7)
            score, s_map = compute_patch_anomaly_map(p, coreset_bank, feat_dim_grid=(28, 28), blur_filter=self.blur)
            
            image_preds.append(score)
            image_labels.append(s["label"])
            pixel_preds.extend(s_map.flatten())
            gt_np = s["gt"].squeeze().numpy()
            pixel_labels.extend(gt_np.flatten())
            predictions.append(s_map)
            gts.append(gt_np)
            
        i_auroc = roc_auc_score(image_labels, image_preds)
        p_auroc = roc_auc_score(pixel_labels, pixel_preds)
        au_pro, _ = calculate_au_pro(gts, predictions)
        
        return {
            "category": category,
            "image_rocauc": round(float(i_auroc), 4),
            "pixel_rocauc": round(float(p_auroc), 4),
            "au_pro": round(float(au_pro), 4)
        }

    def evaluate_h2_xyz(self, category):
        """H2: Point-MAE XYZ Memory Bank Baseline."""
        train_samples = self.load_category_samples(category, "train")
        test_samples = self.load_category_samples(category, "test")
        
        # Build Train Bank
        xyz_patches = [s["xyz_patch"] for s in train_samples] # each (3136, 1152)
        patch_lib = torch.cat(xyz_patches, dim=0)
        
        mean = torch.mean(patch_lib)
        std = torch.std(patch_lib)
        patch_lib = (patch_lib - mean) / (std + 1e-7)
        
        # Coreset
        coreset_idx = get_coreset_indices(patch_lib, f_coreset=0.1, cache_tag=f"{category}_xyz")
        coreset_bank = patch_lib[coreset_idx]
        
        image_preds, image_labels = [], []
        pixel_preds, pixel_labels = [], []
        gts, predictions = [], []
        
        for s in test_samples:
            p = (s["xyz_patch"] - mean) / (std + 1e-7)
            score, s_map = compute_patch_anomaly_map(p, coreset_bank, feat_dim_grid=(56, 56), blur_filter=self.blur)
            
            image_preds.append(score)
            image_labels.append(s["label"])
            pixel_preds.extend(s_map.flatten())
            gt_np = s["gt"].squeeze().numpy()
            pixel_labels.extend(gt_np.flatten())
            predictions.append(s_map)
            gts.append(gt_np)
            
        i_auroc = roc_auc_score(image_labels, image_preds)
        p_auroc = roc_auc_score(pixel_labels, pixel_preds)
        au_pro, _ = calculate_au_pro(gts, predictions)
        
        return {
            "category": category,
            "image_rocauc": round(float(i_auroc), 4),
            "pixel_rocauc": round(float(p_auroc), 4),
            "au_pro": round(float(au_pro), 4)
        }

    def evaluate_h3_multimodal(self, category):
        """H3: Simple Multimodal Baselines (H3A: Calibrated 50/50, H3B: Max, H3C: Aligned Concat)."""
        train_samples = self.load_category_samples(category, "train")
        test_samples = self.load_category_samples(category, "test")
        
        # 1. RGB Bank
        rgb_patches = [s["rgb_patch"] for s in train_samples]
        rgb_lib = torch.cat(rgb_patches, dim=0)
        rgb_mean, rgb_std = torch.mean(rgb_lib), torch.std(rgb_lib)
        rgb_lib = (rgb_lib - rgb_mean) / (rgb_std + 1e-7)
        rgb_coreset_idx = get_coreset_indices(rgb_lib, f_coreset=0.1, cache_tag=f"{category}_rgb")
        rgb_bank = rgb_lib[rgb_coreset_idx]
        
        # 2. XYZ Bank
        xyz_patches = [s["xyz_patch"] for s in train_samples]
        xyz_lib = torch.cat(xyz_patches, dim=0)
        xyz_mean, xyz_std = torch.mean(xyz_lib), torch.std(xyz_lib)
        xyz_lib = (xyz_lib - xyz_mean) / (xyz_std + 1e-7)
        xyz_coreset_idx = get_coreset_indices(xyz_lib, f_coreset=0.1, cache_tag=f"{category}_xyz")
        xyz_bank = xyz_lib[xyz_coreset_idx]
        
        # 3. Concat Bank (H3C)
        concat_patches = [s["combined_patch"] for s in train_samples] # (3136, 1920)
        concat_lib = torch.cat(concat_patches, dim=0)
        concat_mean, concat_std = torch.mean(concat_lib), torch.std(concat_lib)
        concat_lib = (concat_lib - concat_mean) / (concat_std + 1e-7)
        concat_coreset_idx = get_coreset_indices(concat_lib, f_coreset=0.1, cache_tag=f"{category}_concat")
        concat_bank = concat_lib[concat_coreset_idx]
        
        # Scoring lists
        preds_h3a_img, preds_h3b_img, preds_h3c_img = [], [], []
        preds_h3a_pix, preds_h3b_pix, preds_h3c_pix = [], [], []
        maps_h3a, maps_h3b, maps_h3c = [], [], []
        gts, img_labels, pix_labels = [], [], []
        
        for s in test_samples:
            # RGB score & map
            p_rgb = (s["rgb_patch"] - rgb_mean) / (rgb_std + 1e-7)
            s_rgb, map_rgb = compute_patch_anomaly_map(p_rgb, rgb_bank, feat_dim_grid=(28, 28), blur_filter=self.blur)
            
            # XYZ score & map
            p_xyz = (s["xyz_patch"] - xyz_mean) / (xyz_std + 1e-7)
            s_xyz, map_xyz = compute_patch_anomaly_map(p_xyz, xyz_bank, feat_dim_grid=(56, 56), blur_filter=self.blur)
            
            # Concat score & map (H3C)
            p_concat = (s["combined_patch"] - concat_mean) / (concat_std + 1e-7)
            s_concat, map_concat = compute_patch_anomaly_map(p_concat, concat_bank, feat_dim_grid=(56, 56), blur_filter=self.blur)
            
            # H3A: Calibrated 50/50 fusion
            s_h3a = 0.5 * s_rgb + 0.5 * s_xyz
            map_h3a = 0.5 * map_rgb + 0.5 * map_xyz
            
            # H3B: Max fusion
            s_h3b = max(s_rgb, s_xyz)
            map_h3b = np.maximum(map_rgb, map_xyz)
            
            # Record
            preds_h3a_img.append(s_h3a)
            preds_h3b_img.append(s_h3b)
            preds_h3c_img.append(s_concat)
            
            preds_h3a_pix.extend(map_h3a.flatten())
            preds_h3b_pix.extend(map_h3b.flatten())
            preds_h3c_pix.extend(map_concat.flatten())
            
            maps_h3a.append(map_h3a)
            maps_h3b.append(map_h3b)
            maps_h3c.append(map_concat)
            
            gt_np = s["gt"].squeeze().numpy()
            gts.append(gt_np)
            img_labels.append(s["label"])
            pix_labels.extend(gt_np.flatten())
            
        res_h3a = {
            "category": category,
            "variant": "H3A_Calibrated_50_50",
            "image_rocauc": round(float(roc_auc_score(img_labels, preds_h3a_img)), 4),
            "pixel_rocauc": round(float(roc_auc_score(pix_labels, preds_h3a_pix)), 4),
            "au_pro": round(float(calculate_au_pro(gts, maps_h3a)[0]), 4)
        }
        res_h3b = {
            "category": category,
            "variant": "H3B_Max_Fusion",
            "image_rocauc": round(float(roc_auc_score(img_labels, preds_h3b_img)), 4),
            "pixel_rocauc": round(float(roc_auc_score(pix_labels, preds_h3b_pix)), 4),
            "au_pro": round(float(calculate_au_pro(gts, maps_h3b)[0]), 4)
        }
        res_h3c = {
            "category": category,
            "variant": "H3C_Concat_Fusion",
            "image_rocauc": round(float(roc_auc_score(img_labels, preds_h3c_img)), 4),
            "pixel_rocauc": round(float(roc_auc_score(pix_labels, preds_h3c_pix)), 4),
            "au_pro": round(float(calculate_au_pro(gts, maps_h3c)[0]), 4)
        }
        return res_h3a, res_h3b, res_h3c

def run_all(experiment="all", cache_root="/opt/xmv_h_fast/canonical", out_dir="/mnt/c/Users/CharanOp/xmv-ad/experiments/high_accuracy/tables"):
    os.makedirs(out_dir, exist_ok=True)
    evaluator = CachedEvaluator(cache_root=cache_root)
    
    if experiment in ["all", "H1"]:
        print("\n==========================================")
        print("RUNNING H1: DINO RGB MEMORY BASELINE")
        print("==========================================")
        h1_rows = []
        for cat in CATEGORIES:
            t0 = time.time()
            res = evaluator.evaluate_h1_rgb(cat)
            t1 = time.time()
            print(f"H1 [{cat}] -> I-AUROC: {res['image_rocauc']}, P-AUROC: {res['pixel_rocauc']}, AUPRO: {res['au_pro']} ({t1-t0:.2f}s)")
            h1_rows.append(res)
            
        mean_row = {
            "category": "MEAN",
            "image_rocauc": round(float(np.mean([r["image_rocauc"] for r in h1_rows])), 4),
            "pixel_rocauc": round(float(np.mean([r["pixel_rocauc"] for r in h1_rows])), 4),
            "au_pro": round(float(np.mean([r["au_pro"] for r in h1_rows])), 4)
        }
        h1_rows.append(mean_row)
        print(f"\n---> H1 MEAN: I-AUROC={mean_row['image_rocauc']}, P-AUROC={mean_row['pixel_rocauc']}, AUPRO={mean_row['au_pro']}")
        
        with open(os.path.join(out_dir, "H1_DINO_RGB.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "image_rocauc", "pixel_rocauc", "au_pro"])
            writer.writeheader()
            writer.writerows(h1_rows)
            
    if experiment in ["all", "H2"]:
        print("\n==========================================")
        print("RUNNING H2: POINT-MAE XYZ MEMORY BASELINE")
        print("==========================================")
        h2_rows = []
        for cat in CATEGORIES:
            t0 = time.time()
            res = evaluator.evaluate_h2_xyz(cat)
            t1 = time.time()
            print(f"H2 [{cat}] -> I-AUROC: {res['image_rocauc']}, P-AUROC: {res['pixel_rocauc']}, AUPRO: {res['au_pro']} ({t1-t0:.2f}s)")
            h2_rows.append(res)
            
        mean_row = {
            "category": "MEAN",
            "image_rocauc": round(float(np.mean([r["image_rocauc"] for r in h2_rows])), 4),
            "pixel_rocauc": round(float(np.mean([r["pixel_rocauc"] for r in h2_rows])), 4),
            "au_pro": round(float(np.mean([r["au_pro"] for r in h2_rows])), 4)
        }
        h2_rows.append(mean_row)
        print(f"\n---> H2 MEAN: I-AUROC={mean_row['image_rocauc']}, P-AUROC={mean_row['pixel_rocauc']}, AUPRO={mean_row['au_pro']}")
        
        with open(os.path.join(out_dir, "H2_PointMAE_XYZ.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "image_rocauc", "pixel_rocauc", "au_pro"])
            writer.writeheader()
            writer.writerows(h2_rows)
            
    if experiment in ["all", "H3"]:
        print("\n==========================================")
        print("RUNNING H3: MULTIMODAL BASELINES (H3A, H3B, H3C)")
        print("==========================================")
        h3a_rows, h3b_rows, h3c_rows = [], [], []
        for cat in CATEGORIES:
            t0 = time.time()
            r_a, r_b, r_c = evaluator.evaluate_h3_multimodal(cat)
            t1 = time.time()
            print(f"H3 [{cat}] ({t1-t0:.2f}s):")
            print(f"   H3A (50/50): I-AUROC={r_a['image_rocauc']}, P-AUROC={r_a['pixel_rocauc']}, AUPRO={r_a['au_pro']}")
            print(f"   H3B (Max):   I-AUROC={r_b['image_rocauc']}, P-AUROC={r_b['pixel_rocauc']}, AUPRO={r_b['au_pro']}")
            print(f"   H3C (Concat):I-AUROC={r_c['image_rocauc']}, P-AUROC={r_c['pixel_rocauc']}, AUPRO={r_c['au_pro']}")
            h3a_rows.append(r_a)
            h3b_rows.append(r_b)
            h3c_rows.append(r_c)
            
        for name, rows in [("H3A_Calibrated_50_50", h3a_rows), ("H3B_Max_Fusion", h3b_rows), ("H3C_Concat_Fusion", h3c_rows)]:
            mean_row = {
                "category": "MEAN",
                "variant": name,
                "image_rocauc": round(float(np.mean([r["image_rocauc"] for r in rows])), 4),
                "pixel_rocauc": round(float(np.mean([r["pixel_rocauc"] for r in rows])), 4),
                "au_pro": round(float(np.mean([r["au_pro"] for r in rows])), 4)
            }
            rows.append(mean_row)
            print(f"\n---> {name} MEAN: I-AUROC={mean_row['image_rocauc']}, P-AUROC={mean_row['pixel_rocauc']}, AUPRO={mean_row['au_pro']}")
            
        with open(os.path.join(out_dir, "H3_Multimodal.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "variant", "image_rocauc", "pixel_rocauc", "au_pro"])
            writer.writeheader()
            writer.writerows(h3a_rows + h3b_rows + h3c_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["all", "H1", "H2", "H3"])
    parser.add_argument("--cache", default="/opt/xmv_h_fast/canonical")
    parser.add_argument("--out-dir", default="/mnt/c/Users/CharanOp/xmv-ad/experiments/high_accuracy/tables")
    args = parser.parse_args()
    
    run_all(experiment=args.experiment, cache_root=args.cache, out_dir=args.out_dir)
