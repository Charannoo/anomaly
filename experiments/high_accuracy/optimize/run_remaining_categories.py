#!/usr/bin/env python3
"""Run remaining H0-R categories using the optimized pipeline.

After cable_gland finishes, this script runs all remaining categories
with the optimized pipeline.

Categories to run (excluding bagel and cable_gland):
carrot, cookie, dowel, foam, peach, potato, rope, tire
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

CATEGORIES = ["carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]
RESULTS_DIR = Path("/home/hiran/m3dm_work/results")
LOG_DIR = Path("/home/hiran/m3dm_work/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_optimized_category(class_name, num_workers=0, dino_batch_size=1, use_cache=True):
    """Run a single category with the optimized pipeline."""
    cmd = [
        sys.executable, "/home/hiran/m3dm_work/run_stage1_optimized.py",
        "--class_name", class_name,
        "--method_name", "DINO+Point_MAE",
        "--memory_bank", "multiple",
        "--rgb_backbone_name", "vit_base_patch8_224_dino",
        "--xyz_backbone_name", "Point_MAE",
        "--dataset_path", "/opt/mvtec3d",
        "--img_size", "224",
        "--group_size", "128",
        "--num_group", "256",
        "--f_coreset", "0.1",
        "--coreset_eps", "0.9",
        "--max_sample", "400",
        "--num_workers", str(num_workers),
        "--dino_batch_size", str(dino_batch_size),
        "--use_cache", str(use_cache).lower(),
        "--save_feature",
        "--save_feature_path", "/home/hiran/m3dm_work/feature_cache",
        "--out_csv", str(RESULTS_DIR / "optimized_results.csv"),
        "--stage", "all",
    ]
    
    log_file = LOG_DIR / f"opt_{class_name}.log"
    env = {"PYTHONUNBUFFERED": "1"}
    
    print(f"Starting {class_name} at {datetime.now(timezone.utc).isoformat()}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**env, "PATH": "/opt/m3dm/bin:/usr/bin:/bin"},
    )
    
    with open(log_file, "w") as f:
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(f"[{class_name}] {line.rstrip()}")
    
    process.wait()
    elapsed = time.perf_counter() - start_time
    
    return {
        "class": class_name,
        "returncode": process.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def main():
    """Run all remaining categories sequentially."""
    # First check which categories are already complete
    opt_csv = RESULTS_DIR / "optimized_results.csv"
    completed = set()
    if opt_csv.exists():
        import pandas as pd
        df = pd.read_csv(opt_csv)
        completed = set(df["class"].tolist()) if "class" in df.columns else set()
    
    # Also check stage1 results
    stage1_csv = Path("/opt/tmp/stage1_results.csv")
    if stage1_csv.exists():
        import pandas as pd
        df = pd.read_csv(stage1_csv)
        completed.update(df["class"].tolist())
    
    remaining = [c for c in CATEGORIES if c not in completed]
    print(f"Completed: {completed}")
    print(f"Remaining: {remaining}")
    
    results = []
    for cat in remaining:
        result = run_optimized_category(cat)
        results.append(result)
        print(f"\n{cat} completed in {result['elapsed_seconds']:.1f}s")
        print(f"Status: {'SUCCESS' if result['returncode'] == 0 else 'FAILED'}")
        
        if result["returncode"] != 0:
            print(f"ERROR: {cat} failed. Check log at {LOG_DIR / f'opt_{cat}.log'}")
            break  # Stop on failure to avoid cascading errors
    
    # Write summary
    with open(RESULTS_DIR / "remaining_categories_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll remaining categories processed.")
    print(f"Results saved to {RESULTS_DIR / 'remaining_categories_summary.json'}")

if __name__ == "__main__":
    main()
