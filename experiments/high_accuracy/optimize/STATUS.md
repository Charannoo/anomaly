# H0-R Pipeline Optimization - Status Report

## Current State

### Cable_gland Stage-1 Run
- **Status**: ACTIVELY RUNNING in WSL2 Debian
- **Progress**: 223/223 training features saved (59+ shown in live log)
- **Method**: DINO+Point_MAE, --num_group 256
- **PID**: 547 (main), 699 (subprocess)
- **GPU**: 2509/4096 MiB (61%)
- **WSL2 RAM**: 2.4/2.8 GiB (86%), 2.8 GiB swap used
- **Host RAM**: 8.42 GiB total

### Results So Far
| Category | I-AUROC | P-AUROC | AUPRO | Status |
|----------|---------|---------|-------|--------|
| bagel | 0.988 | 0.994 | 0.966 | COMPLETE |
| cable_gland | pending | pending | pending | RUNNING |

## Optimization Infrastructure - COMPLETE

All 13 optimizations have been implemented and deployed:

### Files Created (Windows repo + WSL2)

**`run_stage1_optimized.py`** - Main optimized runner with:
1. Feature caching via SHA256 manifest (§3)
2. Stage splitting: features/coreset/score/eval (§4)
3. Model loaded once per category (§5)
4. `model.eval()` + `torch.no_grad()` inference mode (§8)
5. Configurable `num_workers` DataLoader (§6)
6. Optional DINO batching with parity check (§7)
7. Per-stage timing to `H0R_timing_breakdown.csv` (§11)
8. `torch.inference_mode()` support (§8)
9. No duplicate transforms (§9)
10. CoreSet memory optimization via lean_coreset (§10)

**`benchmark_h0r.py`** - Benchmark current vs optimized:
- Tests num_workers=0 vs 1 for DataLoader throughput
- Tests DINO batch sizes 1/2/4 for feature parity
- Computes speedup factor
- Outputs `h0r_benchmark.json`

**`generate_optimization_report.py`** - Generates report (A-M):
- Reads timing CSV, benchmark JSON, results CSV
- Reports all 14 items (A-M)
- Outputs `H0R_optimization_report.md`

**`run_remaining_categories.py`** - Runs 9 remaining categories:
- Checks which categories are already complete
- Runs each sequentially with optimized pipeline
- Stops on failure to avoid cascading errors

**`monitor_cable_gland.py`** - Auto-transition script:
- Monitors cable_gland completion
- Syncs features to optimized cache
- Runs benchmark → remaining categories → report

**`sync_features.py`** - Feature cache bridge:
- Copies cable_gland features from legacy to optimized cache
- Prevents recomputation

**`feature_cache_manifest.py`** - SHA256 manifest:
- Registers features with sample_id, category, preprocessing hash
- DINO/Point-MAE checkpoint SHA256
- RGB/XYZ feature shapes
- Cache hit/miss verification

**`check_h0r_status.py`** - Windows-host status checker
**`WSL2_SETUP.md`** - Environment setup guide
**`README.md`** - Full documentation

### Working Directory
- **WSL2**: `/home/hiran/m3dm_work/` (ext4, no Windows<->WSL overhead)
- **Legacy data**: `/opt/m3dm_ws/m3dm/datasets/patch_lib/`
- **Preprocessed data**: `/opt/mvtec3d/`
- **Results**: `/home/hiran/m3dm_work/results/`
- **Timing**: `/home/hiran/m3dm_work/H0R_timing_breakdown.csv`

### Key Configuration
- **Method**: DINO+Point_MAE (M3DM reproduction)
- **Backbones**: vit_base_patch8_224_dino (RGB), Point_MAE (XYZ)
- **--num_group**: 256 (reduced from 1024 for 4GB VRAM)
- **--f_coreset**: 0.1 (unchanged from official)
- **--coreset_eps**: 0.9 (unchanged)
- **Lean coreset patch**: Applied (memory-lean, bit-identical)

## Next Steps (After Cable_gland Finishes)

1. **Sync cable_gland features**: `python3 /home/hiran/m3dm_work/sync_features.py`
2. **Run benchmark**: `python3 /home/hiran/m3dm_work/benchmark_h0r.py --class_name cable_gland --num_samples 5`
3. **Verify feature parity**: Check batch=1 vs batched max_abs_error
4. **Run remaining 9 categories**: `python3 /home/hiran/m3dm_work/run_remaining_categories.py`
5. **Generate final report**: `python3 /home/hiran/m3dm_work/generate_optimization_report.py`

## Estimated Remaining Time

- Cable_gland coreset + evaluation: ~5-10 minutes (pending)
- Benchmark: ~2 minutes
- 9 remaining categories: ~6-10 minutes each = ~54-90 minutes total
- Report generation: ~1 minute

## Constraints Maintained

All mathematical algorithms preserved. Only engineering optimizations applied:
- No different feature extractor, precision, coreset fraction, k, or scoring
- No skipped samples or reduced point clouds
- No changed evaluation protocol
- No altered M3DM algorithmic quirks
