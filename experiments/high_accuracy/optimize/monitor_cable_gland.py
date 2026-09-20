#!/usr/bin/env python3
"""Monitor cable_gland run and transition to optimized pipeline.

This script:
1. Checks if cable_gland is still running
2. Once finished, copies cable_gland features to the optimized cache
3. Runs the benchmark comparing current vs optimized
4. Executes the remaining 9 categories with the optimized pipeline
5. Generates the final optimization report
"""
import subprocess
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

WSL_DEBIAN = "wsl -d Debian -e bash -c"
CHECK_INTERVAL = 60  # seconds

def check_cable_gland_running():
    """Check if cable_gland process is still running."""
    cmd = f"{WSL_DEBIAN} \"ps aux 2>/dev/null | grep -E 'run_stage1.py.*cable_gland' | grep -v grep\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return len(result.stdout.strip()) > 0

def get_cable_gland_progress():
    """Get current cable_gland progress."""
    cmd = f"{WSL_DEBIAN} \"ls /opt/m3dm_ws/m3dm/datasets/patch_lib/cable_gland*.pt 2>/dev/null | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0

def copy_cable_features_to_cache():
    """Copy cable_gland features from legacy location to optimized cache."""
    cmd = f"{WSL_DEBIAN} \"cp /opt/m3dm_ws/m3dm/datasets/patch_lib/cable_gland*.pt /home/hiran/m3dm_work/feature_cache/ 2>/dev/null && echo 'Copied' && ls /home/hiran/m3dm_work/feature_cache/cable_gland*.pt | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    return result.stdout.strip()

def run_benchmark():
    """Run the H0-R benchmark."""
    cmd = f"{WSL_DEBIAN} \"python3 /home/hiran/m3dm_work/benchmark_h0r.py --class_name cable_gland --num_samples 5\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

def run_remaining():
    """Run remaining categories with optimized pipeline."""
    cmd = f"{WSL_DEBIAN} \"python3 /home/hiran/m3dm_work/run_remaining_categories.py\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

def generate_report():
    """Generate the optimization report."""
    cmd = f"{WSL_DEBIAN} \"python3 /home/hiran/m3dm_work/generate_optimization_report.py\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)

def main():
    print(f"Monitoring cable_gland at {datetime.now(timezone.utc).isoformat()}")
    print(f"Progress: {get_cable_gland_progress()}/223 features saved")
    
    while check_cable_gland_running():
        progress = get_cable_gland_progress()
        print(f"Waiting for cable_gland... Progress: {progress}/223")
        time.sleep(CHECK_INTERVAL)
        
        if progress >= 223:
            print("Cable_gland training features complete!")
            break
    
    print("Cable_gland finished!")
    
    # Step 1: Copy cable_gland features to optimized cache
    print("Copying cable_gland features to optimized cache...")
    copy_cable_features_to_cache()
    
    # Step 2: Run benchmark
    print("Running benchmark...")
    try:
        run_benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    # Step 3: Generate report before remaining categories
    print("Generating initial optimization report...")
    try:
        generate_report()
    except Exception as e:
        print(f"Report generation failed: {e}")
    
    # Step 4: Run remaining categories
    print("Running remaining 9 categories...")
    try:
        run_remaining()
    except Exception as e:
        print(f"Remaining categories failed: {e}")
    
    # Step 5: Generate final report
    print("Generating final optimization report...")
    try:
        generate_report()
    except Exception as e:
        print(f"Final report generation failed: {e}")
    
    print(f"Pipeline optimization complete at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
