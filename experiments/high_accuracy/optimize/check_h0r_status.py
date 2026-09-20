#!/usr/bin/env python3
"""Windows-host monitoring script for H0-R optimization pipeline.

Run this from Windows cmd to check WSL2 status:
    python check_h0r_status.py

Usage:
    python check_h0r_status.py           # Check current status
    python check_h0r_status.py --full    # Full details
    python check_h0r_status.py --report  # Generate report
"""
import subprocess
import sys
import json
import argparse
from pathlib import Path

def run_wsl(cmd):
    """Run a command in WSL2 Debian."""
    result = subprocess.run(
        f'wsl -d Debian -e bash -c "{cmd}"',
        shell=True, capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()

def check_cable_gland():
    """Check if cable_gland is still running."""
    procs = run_wsl("ps aux 2>/dev/null | grep 'run_stage1.py.*cable_gland' | grep -v grep")
    return len(procs) > 0, procs

def get_progress():
    """Get cable_gland training progress."""
    count = run_wsl("ls /opt/m3dm_ws/m3dm/datasets/patch_lib/cable_gland*.pt 2>/dev/null | wc -l")
    return int(count) if count.strip().isdigit() else 0

def get_results():
    """Get stage1 results CSV."""
    csv = run_wsl("cat /opt/tmp/stage1_results.csv")
    return csv

def get_gpu_status():
    """Get GPU memory status."""
    return run_wsl("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null")

def get_memory():
    """Get WSL2 memory status."""
    return run_wsl("free -h 2>/dev/null")

def get_timing_csv():
    """Get timing breakdown CSV."""
    return run_wsl("cat /home/hiran/m3dm_work/H0R_timing_breakdown.csv 2>/dev/null")

def get_benchmark():
    """Get benchmark results."""
    path = Path(__file__).parent / "experiments/high_accuracy/results/h0r_benchmark.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def print_status(full=False):
    """Print current H0-R status."""
    running, procs = check_cable_gland()
    progress = get_progress()
    gpu = get_gpu_status()
    mem = get_memory()
    results = get_results()
    timing = get_timing_csv()
    
    print("=" * 60)
    print("  H0-R OPTIMIZATION PIPELINE STATUS")
    print("=" * 60)
    print(f"\n  Cable_gland running: {running}")
    print(f"  Training features saved: {progress}/223")
    if procs:
        print(f"  Process(es): {procs[:2]}")
    print(f"\n  GPU: {gpu}")
    print(f"\n  Memory: {mem}")
    print(f"\n  Stage-1 Results:\n{results}")
    if timing and timing != "stage,seconds,percentage_of_total":
        print(f"\n  Timing Breakdown:\n{timing}")
    
    benchmark = get_benchmark()
    if benchmark:
        print(f"\n  Benchmark Results:")
        print(f"    Original throughput: {benchmark.get('original_throughput_samples_per_sec', 'N/A')} samples/sec")
        print(f"    Optimized throughput: {benchmark.get('optimized_throughput_samples_per_sec', 'N/A')} samples/sec")
        print(f"    Speedup: {benchmark.get('speedup_factor', 'N/A')}x")
    
    print(f"\n  Working directory: /home/hiran/m3dm_work/")
    print(f"  Scripts: run_stage1_optimized.py, benchmark_h0r.py, etc.")
    print(f"\n  Next step: Run benchmark after cable_gland finishes")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    print_status(full=args.full)
