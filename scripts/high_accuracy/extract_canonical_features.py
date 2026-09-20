#!/usr/bin/env python3
"""Canonical Feature Extractor for XMV-AD-H.

Extracts frozen DINO ViT-B/8 and Point-MAE features once for all 10 MVTec 3D-AD
categories with atomic caching, resume support, and high-throughput memory-lean execution.
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
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Ensure M3DM modules can be imported
sys.path.insert(0, '/opt/m3dm_ws/m3dm')
if os.path.exists('/opt/m3dm_ws/m3dm'):
    os.chdir('/opt/m3dm_ws/m3dm')

from utils.mvtec3d_util import (
    read_tiff_organized_pc,
    organized_pc_to_depth_map,
    resize_organized_pc,
    organized_pc_to_unorganized_pc
)
from models.models import Model
from models.pointnet2_utils import square_distance, index_points

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire"
]

RGB_SIZE = 224

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def chunked_interpolating_points(xyz1, xyz2, points2, chunk_size=30000):
    xyz1_p = xyz1.permute(0, 2, 1) # B, N, C
    xyz2_p = xyz2.permute(0, 2, 1) # B, S, C
    points2_p = points2.permute(0, 2, 1) # B, S, D
    B, N, C = xyz1_p.shape
    _, S, D = points2_p.shape
    
    out_parts = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xyz1_chunk = xyz1_p[:, start:end, :]
        dists = square_distance(xyz1_chunk, xyz2_p)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        pts_chunk = torch.sum(index_points(points2_p, idx) * weight.view(B, end - start, 3, 1), dim=2)
        out_parts.append(pts_chunk)
    interpolated_points = torch.cat(out_parts, dim=1).permute(0, 2, 1)
    return interpolated_points

def robust_open_image(path, retries=5):
    for i in range(retries):
        try:
            with Image.open(path) as img:
                return img.convert('RGB')
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(0.5)

def robust_read_tiff(path, retries=5):
    for i in range(retries):
        try:
            return read_tiff_organized_pc(path)
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(0.5)

class CanonicalExtractor:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Initializing Canonical Extractor on device: {self.device}")
        
        # Load DINO + Point-MAE model once
        self.model = Model(
            device=self.device,
            rgb_backbone_name='vit_base_patch8_224_dino',
            xyz_backbone_name='Point_MAE',
            group_size=128,
            num_group=1024
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.avg_pool = torch.nn.AvgPool2d(3, stride=1)
        self.resize_pool = torch.nn.AdaptiveAvgPool2d((56, 56))
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    @torch.inference_mode()
    def extract_sample(self, rgb_path, tiff_path, gt_path=None, label=0, sample_meta=None):
        t_io0 = time.time()
        img = robust_open_image(rgb_path)
        rgb_tensor = self.rgb_transform(img).unsqueeze(0).to(self.device)
        
        organized_pc = robust_read_tiff(tiff_path)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=RGB_SIZE, target_width=RGB_SIZE)
        organized_pc_np = resized_organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        # If points < 1024, pad with repeats to prevent Point-MAE crash
        if len(nonzero_indices) < 1024:
            if len(nonzero_indices) == 0:
                unorganized_pc_valid = np.zeros((1024, 3), dtype=np.float32)
                nonzero_indices = np.arange(1024)
            else:
                rep = int(np.ceil(1024 / len(nonzero_indices)))
                unorganized_pc_valid = np.tile(unorganized_pc[nonzero_indices, :], (rep, 1))[:1024]
        else:
            unorganized_pc_valid = unorganized_pc[nonzero_indices, :]
            
        unorganized_pc_no_zeros = torch.tensor(
            unorganized_pc_valid,
            dtype=torch.float32
        ).unsqueeze(dim=0).permute(0, 2, 1).to(self.device)
        t_io = time.time() - t_io0
        
        # DINO forward
        t_dino0 = time.time()
        rgb_feat = self.model.forward_rgb_features(rgb_tensor) # (1, 768, 28, 28)
        rgb_patch = rgb_feat.reshape(rgb_feat.shape[1], -1).T.cpu() # (784, 768)
        rgb_patch_resized = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1) # (3136, 768)
        t_dino = time.time() - t_dino0
        
        # Point-MAE forward
        t_pmae0 = time.time()
        xyz_features, center, ori_idx, center_idx = self.model.xyz_backbone(unorganized_pc_no_zeros.contiguous())
        interpolated_pc = chunked_interpolating_points(
            unorganized_pc_no_zeros.contiguous(),
            center.permute(0, 2, 1),
            xyz_features
        ) # (1, 1152, N_points)
        
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], RGB_SIZE * RGB_SIZE), dtype=interpolated_pc.dtype, device=self.device)
        if len(nonzero_indices) > 0:
            xyz_patch_full[:, :, nonzero_indices] = interpolated_pc[:, :, :len(nonzero_indices)]
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], RGB_SIZE, RGB_SIZE)
        xyz_patch_full_resized = self.resize_pool(self.avg_pool(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T.cpu() # (3136, 1152)
        t_pmae = time.time() - t_pmae0
        
        # GT mask
        if gt_path is not None and os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('L')
            gt_tensor = self.gt_transform(gt_img)
            gt_tensor = torch.where(gt_tensor > 0.5, 1.0, 0.0)[:1]
        else:
            gt_tensor = torch.zeros((1, RGB_SIZE, RGB_SIZE), dtype=torch.float32)
            
        combined_patch = torch.cat([xyz_patch, rgb_patch_resized], dim=1) # (3136, 1920)
        
        t_ser0 = time.time()
        sample_dict = {
            "sample_id": sample_meta.get("sample_id"),
            "category": sample_meta.get("category"),
            "split": sample_meta.get("split"),
            "defect_type": sample_meta.get("defect_type"),
            "rgb_patch": rgb_patch.to(torch.float32),          # (784, 768)
            "xyz_patch": xyz_patch.to(torch.float32),          # (3136, 1152)
            "rgb_patch_resized": rgb_patch_resized.to(torch.float32), # (3136, 768)
            "combined_patch": combined_patch.to(torch.float32), # (3136, 1920)
            "nonzero_indices": nonzero_indices,
            "gt": gt_tensor.to(torch.float32),
            "label": label,
            "rgb_path": rgb_path,
            "tiff_path": tiff_path
        }
        t_ser = time.time() - t_ser0
        
        timing = {
            "io": t_io,
            "dino": t_dino,
            "pmae": t_pmae,
            "serialize": t_ser,
            "total": t_io + t_dino + t_pmae + t_ser
        }
        return sample_dict, timing


def run_extraction(data_root="/opt/mvtec3d", out_root="/opt/xmv_h_fast/canonical", sync_root="/mnt/c/Users/CharanOp/xmv-ad/experiments/high_accuracy/features/canonical", categories="all", resume=True):
    if categories == "all":
        target_cats = CATEGORIES
    else:
        target_cats = [c.strip() for c in categories.split(",") if c.strip() in CATEGORIES]
        
    print(f"Target Categories ({len(target_cats)}): {target_cats}")
    os.makedirs(out_root, exist_ok=True)
    
    # Checkpoint provenance
    pointmae_ckpt = "/opt/m3dm_ws/m3dm/checkpoints/pointmae_pretrain.pth"
    dino_ckpt = "/root/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth"
    pmae_hash = sha256_file(pointmae_ckpt) if os.path.exists(pointmae_ckpt) else "unknown"
    dino_hash = sha256_file(dino_ckpt) if os.path.exists(dino_ckpt) else "unknown"
    
    metadata = {
        "dataset": "MVTec-3D-AD (canonical preprocessed)",
        "dino_checkpoint_sha256": dino_hash,
        "pointmae_checkpoint_sha256": pmae_hash,
        "rgb_shape": [784, 768],
        "xyz_shape": [3136, 1152],
        "combined_shape": [3136, 1920],
        "feature_layers": "DINO ViT-B/8 patch tokens + Point-MAE (1024 groups, 128 group_size)",
        "serialization_format": "torch .pt dictionary per sample",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    extractor = CanonicalExtractor(device="cuda")
    
    manifest_rows = []
    timing_records = []
    
    total_start = time.time()
    total_samples_processed = 0
    total_samples_cached = 0
    
    for cat in target_cats:
        print(f"\n==========================================")
        print(f"Processing Category: {cat}")
        print(f"==========================================")
        
        cat_out_dir = os.path.join(out_root, cat)
        os.makedirs(cat_out_dir, exist_ok=True)
        
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(data_root, cat, split)
            if not os.path.exists(split_dir):
                continue
                
            out_split_dir = os.path.join(cat_out_dir, split)
            os.makedirs(out_split_dir, exist_ok=True)
            
            defect_types = sorted(os.listdir(split_dir))
            for defect_type in defect_types:
                defect_dir = os.path.join(split_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                    
                rgb_files = sorted(glob.glob(os.path.join(defect_dir, "rgb", "*.png")))
                tiff_files = sorted(glob.glob(os.path.join(defect_dir, "xyz", "*.tiff")))
                gt_files = sorted(glob.glob(os.path.join(defect_dir, "gt", "*.png"))) if split == "test" and defect_type != "good" else []
                
                label = 0 if defect_type == "good" else 1
                
                for idx in range(len(rgb_files)):
                    rgb_p = rgb_files[idx]
                    tiff_p = tiff_files[idx]
                    gt_p = gt_files[idx] if idx < len(gt_files) else None
                    sample_name = os.path.splitext(os.path.basename(rgb_p))[0]
                    sample_id = f"{defect_type}_{sample_name}" if defect_type != "good" else sample_name
                    
                    target_file = os.path.join(out_split_dir, f"{sample_id}.pt")
                    tmp_file = os.path.join(out_split_dir, f"{sample_id}.pt.tmp")
                    
                    if resume and os.path.exists(target_file) and os.path.getsize(target_file) > 10000:
                        total_samples_cached += 1
                        manifest_rows.append({
                            "category": cat,
                            "split": split,
                            "sample_id": sample_id,
                            "rgb_cached": True,
                            "xyz_cached": True,
                            "rgb_shape": "(784, 768)",
                            "xyz_shape": "(3136, 1152)",
                            "rgb_checkpoint_hash": dino_hash[:16],
                            "xyz_checkpoint_hash": pmae_hash[:16],
                            "status": "CACHE_HIT"
                        })
                        continue
                            
                    sample_meta = {
                        "sample_id": sample_id,
                        "category": cat,
                        "split": split,
                        "defect_type": defect_type
                    }
                    
                    sample_dict, t_info = extractor.extract_sample(
                        rgb_p, tiff_p, gt_p, label, sample_meta
                    )
                    
                    # Atomic write
                    torch.save(sample_dict, tmp_file)
                    os.replace(tmp_file, target_file)
                    
                    total_samples_processed += 1
                    timing_records.append({
                        "category": cat,
                        "split": split,
                        "sample_id": sample_id,
                        **t_info
                    })
                    
                    manifest_rows.append({
                        "category": cat,
                        "split": split,
                        "sample_id": sample_id,
                        "rgb_cached": True,
                        "xyz_cached": True,
                        "rgb_shape": "(784, 768)",
                        "xyz_shape": "(3136, 1152)",
                        "rgb_checkpoint_hash": dino_hash[:16],
                        "xyz_checkpoint_hash": pmae_hash[:16],
                        "status": "EXTRACTED"
                    })
                    
                    if total_samples_processed % 25 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        avg_t = np.mean([r["total"] for r in timing_records[-25:]])
                        print(f"[{cat} | {split}] Processed {total_samples_processed} new samples (avg {avg_t:.3f}s/sample, {1.0/max(avg_t, 1e-5):.1f} smp/s)", flush=True)
                        
        print(f"Completed category: {cat}", flush=True)
            
    manifest_csv = os.path.join(out_root, "cache_manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "split", "sample_id", "rgb_cached", "xyz_cached",
            "rgb_shape", "xyz_shape", "rgb_checkpoint_hash", "xyz_checkpoint_hash", "status"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Write timing breakdown
    timing_csv = "/mnt/c/Users/CharanOp/xmv-ad/experiments/high_accuracy/tables/FAST_timing_breakdown.csv"
    os.makedirs(os.path.dirname(timing_csv), exist_ok=True)
    if timing_records:
        with open(timing_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "split", "sample_id", "io", "dino", "pmae", "serialize", "total"])
            writer.writeheader()
            writer.writerows(timing_records)
            
    total_elapsed = time.time() - total_start
    print(f"\n==========================================")
    print(f"Canonical Feature Extraction Summary:")
    print(f"Total Cached Samples: {total_samples_cached}")
    print(f"Total Newly Extracted: {total_samples_processed}")
    print(f"Total Wall Time: {total_elapsed:.2f}s")
    if total_samples_processed > 0:
        print(f"Average Extraction Speed: {total_elapsed/total_samples_processed:.3f}s / sample ({total_samples_processed/total_elapsed:.2f} samples/s)")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/opt/mvtec3d", help="Path to preprocessed MVTec 3D dataset")
    parser.add_argument("--output", default="/opt/xmv_h_fast/canonical", help="Path to canonical feature cache")
    parser.add_argument("--sync-to", default="/mnt/c/Users/CharanOp/xmv-ad/experiments/high_accuracy/features/canonical", help="Sync destination")
    parser.add_argument("--categories", default="all", help="Comma-separated categories or 'all'")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from cache")
    args = parser.parse_args()
    
    run_extraction(
        data_root=args.data,
        out_root=args.output,
        sync_root=args.sync_to,
        categories=args.categories,
        resume=not args.no_resume
    )
