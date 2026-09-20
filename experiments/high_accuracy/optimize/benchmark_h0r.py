#!/usr/bin/env python3
"""Benchmark script comparing CURRENT vs OPTIMIZED pipeline for H0-R.

Runs a small category to measure:
- Current pipeline samples/sec
- Optimized pipeline samples/sec
- Speedup factor
- Per-stage timing breakdown

Usage: wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/benchmark_h0r.py --class_name cable_gland --num_samples 5"
"""
import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("/home/hiran/m3dm_work/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_CSV = RESULTS_DIR / "h0r_benchmark_results.csv"

def benchmark_num_workers(class_name, num_samples=5):
    """Benchmark DataLoader with different num_workers settings."""
    import sys
    sys.path.insert(0, "/opt/m3dm_ws/m3dm")
    from dataset import TrainDataset, TestDataset, DataLoader
    
    results = {}
    for nw in [0, 1]:
        train_ds = TrainDataset(class_name=class_name, img_size=224, dataset_path="/opt/mvtec3d")
        dl = DataLoader(dataset=train_ds, batch_size=1, shuffle=False, num_workers=nw, drop_last=False, pin_memory=True)
        
        # Warmup
        for i, (sample, _) in enumerate(dl):
            if i >= 2:
                break
        
        # Benchmark
        t0 = time.perf_counter()
        count = 0
        for i, (sample, _) in enumerate(dl):
            if i >= num_samples:
                break
            count += 1
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        results[nw] = {
            "samples_per_sec": count / elapsed if elapsed > 0 else 0,
            "total_seconds": elapsed,
            "samples": count,
        }
    
    return results

def benchmark_dino_batching(class_name, num_samples=3):
    """Benchmark DINO feature extraction with different batch sizes."""
    import sys
    sys.path.insert(0, "/opt/m3dm_ws/m3dm")
    import timm
    import torch
    
    results = {}
    for bs in [1, 2, 4]:
        model = timm.create_model("vit_base_patch8_224_dino", pretrained=True)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 224, 224))
        
        # Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_samples):
                x = torch.randn(bs, 3, 224, 224)
                _ = model(x)
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        per_sample = elapsed / (num_samples * bs) if bs > 0 else 0
        results[bs] = {
            "total_seconds": elapsed,
            "per_sample_seconds": per_sample,
            "samples_per_sec": bs / per_sample if per_sample > 0 else 0,
        }
    
    del model
    return results

def compare_pipeline(class_name, num_samples=5):
    """Compare current vs optimized pipeline timing."""
    results = {
        "class_name": class_name,
        "num_samples": num_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Benchmark DataLoader
    dl_results = benchmark_num_workers(class_name, num_samples)
    results["dataloader"] = dl_results
    results["chosen_num_workers"] = max(dl_results, key=lambda k: dl_results[k]["samples_per_sec"])
    
    # Benchmark DINO batching
    dino_results = benchmark_dino_batching(class_name, num_samples)
    results["dino_batching"] = dino_results
    results["chosen_dino_batch_size"] = max(dino_results, key=lambda k: dino_results[k]["per_sample_seconds"] < 9999 and dino_results[k]["samples_per_sec"] > 0)
    
    # Estimate speedup
    # Current: num_workers=1, batch=1 (from original run_stage1.py)
    current_throughput = dl_results[1]["samples_per_sec"]
    optimized_throughput = dl_results[results["chosen_num_workers"]]["samples_per_sec"]
    results["original_throughput_samples_per_sec"] = current_throughput
    results["optimized_throughput_samples_per_sec"] = optimized_throughput
    results["speedup_factor"] = optimized_throughput / current_throughput if current_throughput > 0 else 0
    
    # DINO throughput improvement
    results["dino_original_throughput"] = dino_results[1]["samples_per_sec"]
    results["dino_optimized_throughput"] = dino_results[results["chosen_dino_batch_size"]]["samples_per_sec"]
    
    # Save results
    with open(RESULTS_DIR / "h0r_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Write CSV row
    df = pd.DataFrame([{
        "class": class_name,
        "num_samples": num_samples,
        "original_throughput": current_throughput,
        "optimized_throughput": optimized_throughput,
        "speedup_factor": results["speedup_factor"],
        "chosen_num_workers": results["chosen_num_workers"],
        "chosen_dino_batch_size": results["chosen_dino_batch_size"],
        "dino_original_tput": results["dino_original_throughput"],
        "dino_optimized_tput": results["dino_optimized_throughput"],
    }])
    if BENCHMARK_CSV.exists():
        pd.concat([pd.read_csv(BENCHMARK_CSV), df], ignore_index=True).to_csv(BENCHMARK_CSV, index=False)
    else:
        df.to_csv(BENCHMARK_CSV, index=False)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", default="cable_gland", type=str)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--out_dir", default="/home/hiran/m3dm_work/results")
    args = parser.parse_args()
    
    results = compare_pipeline(args.class_name, args.num_samples)
    print(json.dumps(results, indent=2))
