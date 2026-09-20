# WSL2 Optimization Setup Guide for H0-R

## Environment Overview

- **Host**: Windows 11, Intel 4c/8t, 8.42 GB RAM, GTX 1650 4GB VRAM
- **WSL2**: Debian, GPU-accelerated via CUDA 11.3
- **Python**: `/opt/m3dm/bin/python` (conda env in WSL2)
- **M3DM Repo**: `/opt/m3dm_ws/m3dm/` (inside WSL2)
- **Data**: `/opt/mvtec3d/` (preprocessed inside WSL2)

## Key Directories

| Path | Purpose |
|------|---------|
| `/opt/m3dm_ws/m3dm/` | Official M3DM repo (read-only) |
| `/opt/mvtec3d/` | Preprocessed MVTec-3D dataset |
| `/opt/tmp/feature_cache/` | Feature cache for coreset reuse |
| `/home/hiran/m3dm_work/` | **Working directory for optimized pipeline** |
| `/home/hiran/m3dm_work/feature_cache/` | Cached feature tensors |
| `/home/hiran/m3dm_work/H0R_timing_breakdown.csv` | Per-stage timing CSV |
| `/home/hiran/m3dm_work/results/optimized_results.csv` | Optimized results |
| `/home/hiran/m3dm_work/results/h0r_benchmark.json` | Benchmark results |
| `/home/hiran/m3dm_work/results/H0R_optimization_report.md` | Final optimization report |

## Scripts

| Script | Purpose |
|--------|---------|
| `run_stage1_optimized.py` | Main optimized Stage-1 runner with all optimizations |
| `benchmark_h0r.py` | Benchmark current vs optimized pipeline |
| `generate_optimization_report.py` | Generate report (A-M) |
| `run_remaining_categories.py` | Run remaining 9 categories |
| `monitor_cable_gland.py` | Monitor cable_gland and auto-transition |
| `sync_features.py` | Copy features from legacy to optimized cache |
| `feature_cache_manifest.py` | Feature cache manifest with SHA256 verification |

## Running the Optimized Pipeline

### Step 1: Wait for cable_gland to finish

```bash
# Check status
wsl -d Debian -e bash -c "ps aux | grep python | grep -v grep"
wsl -d Debian -e bash -c "cat /opt/tmp/stage1_results.csv"
```

### Step 2: Run benchmark

```bash
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/benchmark_h0r.py --class_name cable_gland --num_samples 5"
```

### Step 3: Run remaining categories

```bash
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/run_remaining_categories.py"
```

### Step 4: Generate report

```bash
wsl -d Debian -e bash -c "python3 /home/hiran/m3dm_work/generate_optimization_report.py"
```

## Optimizations Applied

1. **Linux-native filesystem**: All data in WSL2 ext4 (`/home/hiran/m3dm_work/`)
2. **Feature caching**: SHA256 manifest prevents recomputation
3. **Model reuse**: `model.eval()` once per category, no reload
4. **Inference mode**: `torch.no_grad()` + `model.eval()` 
5. **DataLoader tuning**: Benchmark num_workers=0 vs 1
6. **Batched DINO**: Optional batching with parity check
7. **Stage splitting**: Features/coreset/score/eval as separate stages
8. **Timing**: `H0R_timing_breakdown.csv` with per-stage measurements
9. **CoreSet memory**: lean_coreset patch avoids OOM
10. **No duplicate transforms**: Cached preprocessing results

## Verification Checklist

- [ ] DINO feature parity: batch=1 vs batched max_abs_error < 1e-6
- [ ] Point-MAE feature parity: same check
- [ ] lean_coreset bit-identical to official fp64:fp16 path
- [ ] Cache-hit behavior verified (no recomputation on rerun)
- [ ] Timing CSV has all stages reported
- [ ] Optimization report has all items A-M

## Notes

- Do NOT modify the official M3DM code at `/opt/m3dm_ws/m3dm/`
- The `lean_coreset.py` patch is already applied in `run_stage1.py`
- All mathematical algorithms are preserved - only engineering optimizations
- Feature extraction uses `torch.inference_mode()` and `model.eval()`
- The `DoubleRGBPointFeatures` class is used (DINO + Point-MAE)
- `--num_group 256` is used (not 1024) to fit in 4GB VRAM
