# H0-R Pipeline Optimization

## Overview

Optimization pipeline for the M3DM H0-R (high-accuracy reproduction) stage.
The goal is to reduce wall-clock time for the remaining 9 MVTec-3D categories
while preserving faithful M3DM mathematical reproduction.

## Files

| File | Description |
|------|-------------|
| `run_stage1_optimized.py` | Main optimized Stage-1 runner with all 13 optimizations |
| `benchmark_h0r.py` | Benchmark comparing current vs optimized pipeline |
| `generate_optimization_report.py` | Generates the final report (items A-M) |
| `run_remaining_categories.py` | Runs remaining 9 categories sequentially |
| `monitor_cable_gland.py` | Monitors cable_gland, auto-transitions to optimized pipeline |
| `sync_features.py` | Copies features from legacy to optimized cache |
| `feature_cache_manifest.py` | SHA256-based feature cache manifest |
| `check_h0r_status.py` | Windows-host status checker |
| `check_status.sh` | WSL2 status checker |
| `WSL2_SETUP.md` | Environment setup guide |

## Running from Windows Host

```cmd
:: Check status
python experiments\high_accuracy\optimize\check_h0r_status.py

:: Run benchmark after cable_gland finishes
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/benchmark_h0r.py"

:: Run remaining 9 categories
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/run_remaining_categories.py"

:: Generate final report
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/generate_optimization_report.py"
```

## Running from WSL2

```bash
# Check cable_gland status
wsl -d Debian -e bash -c "cat /opt/tmp/stage1_results.csv"
wsl -d Debian -e bash -c "ls /opt/m3dm_ws/m3dm/datasets/patch_lib/cable_gland*.pt | wc -l"

# Once cable_gland finishes, run benchmark
python3 /home/hiran/m3dm_work/benchmark_h0r.py --class_name cable_gland --num_samples 5

# Run remaining categories
python3 /home/hiran/m3dm_work/run_remaining_categories.py

# Generate report
python3 /home/hiran/m3dm_work/generate_optimization_report.py
```

## Optimizations Applied

1. **Linux-native filesystem**: All working data in `/home/hiran/m3dm_work/` (WSL2 ext4)
2. **Feature caching**: SHA256 manifest prevents recomputation of valid features
3. **Model reuse**: `model.eval()` once per category, no checkpoint reload
4. **Inference mode**: `torch.no_grad()` throughout feature extraction
5. **DataLoader tuning**: Benchmark num_workers=0 vs 1
6. **Batched DINO**: Optional batching wrapper with parity verification
7. **Stage splitting**: Features/coreset/score/eval as separate stages
8. **Timing**: `H0R_timing_breakdown.csv` with per-stage measurements
9. **CoreSet memory**: `lean_coreset.py` patch avoids OOM under 3GB WSL2 cap
10. **No duplicate transforms**: Cached preprocessing results
11. **No unnecessary copies**: Tensor operations optimized
12. **GPU inference mode**: `torch.inference_mode()` + `model.eval()`
13. **Single category per process**: Prevents memory exhaustion

## Key Constraints

- **Do NOT** change M3DM mathematical algorithm
- **Do NOT** change evaluation protocol
- **Do NOT** use different feature extractor, precision, coreset fraction, k, or scoring
- **Do NOT** skip samples or reduce point cloud size
- **Do NOT** begin H1 or any new model

## Hardware

- Host: Windows 11, 8.42 GB RAM, GTX 1650 4GB VRAM
- WSL2: Debian, CUDA 11.3, GPU accessible via nvidia-smi
- Python: `/opt/m3dm/bin/python` (conda)
- Method: DINO+Point_MAE (not full Fusion)
- `--num_group 256` (reduced from 1024 to fit 4GB VRAM)

## Output Files

- `/home/hiran/m3dm_work/H0R_timing_breakdown.csv` - Per-stage timing
- `/home/hiran/m3dm_work/results/optimized_results.csv` - All category results
- `/home/hiran/m3dm_work/results/h0r_benchmark.json` - Benchmark data
- `/home/hiran/m3dm_work/results/H0R_optimization_report.md` - Final report (A-M)
- `/home/hiran/m3dm_work/feature_cache/` - Cached feature tensors
