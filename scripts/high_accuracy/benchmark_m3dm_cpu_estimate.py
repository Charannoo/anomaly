"""H0.8 - CPU micro-benchmark estimating per-sample runtime of dominant M3DM components.

This is NOT an M3DM reproduction. The official M3DM pipeline imports CUDA-only
compiled ops (pointnet2_ops, knn_cuda), so it cannot run on this CPU-only
machine. Here we benchmark the two CPU-runnable dominant per-sample costs to
provide a defensible runtime estimate:

  1. DINO ViT-B/8 (patch 8, 224x224) forward pass  -- RGB feature extraction
  2. nearest-neighbor (coreset-cdist) scoring       -- memory-bank scoring

We also extrapolate distance-scoring cost to the expected M3DM operating sizes
(10% coreset of the training patch memory banks). Everything is labeled an
ESTIMATE, never a reproduction.

Outputs:
  experiments/high_accuracy/runs/cpu_microbenchmark_<ts>.json
"""
import argparse
import json
import platform
import time
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


def bench(fn, warmup=3, reps=12):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "p50_ms": float(np.percentile(arr, 50) * 1e3),
        "p90_ms": float(np.percentile(arr, 90) * 1e3),
        "mean_ms": float(arr.mean() * 1e3),
    }


def bench_dino_vit():
    import timm
    model = timm.create_model("vit_base_patch8_224_dino", pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        return bench(lambda: model(x), warmup=2, reps=10)


def bench_cdist(rows, dim, coreset_rows):
    a = torch.randn(rows, dim)
    b = torch.randn(coreset_rows, dim)
    with torch.no_grad():
        return bench(lambda: torch.cdist(a, b), warmup=1, reps=4)


def extrapolate_ms(measured, coreset_rows, target_rows):
    """Assume cdist time ~ linear in #coreset rows."""
    per_row_ms = measured / coreset_rows
    return per_row_ms * target_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str("experiments/high_accuracy/runs"))
    args = parser.parse_args()

    torch.set_num_threads(8)

    dino = bench_dino_vit()

    # expected M3DM coreset sizes (f_coreset=0.1) for a ~266-train class:
    #   rgb   : 266 samples * 784 patches = 208,544 -> 10% = 20,854 rows, dim 768
    #   xyz   : 266 * 3136 = 833,056 roles     -> 10% = 83,305 rows, dim 1152
    #   fusion: 266 * 3136                      -> 10% = 83,305 rows, dim 1920
    rgb_target = 20854
    xyz_target = 83305
    dim_rgb, dim_xyz, dim_fus = 768, 1152, 1920
    rows_patch_rgb, rows_patch_pt = 784, 3136

    rgb_timings = {}
    for k in (1000, 5000, 10000):
        rgb_timings[k] = bench_cdist(rows_patch_rgb, dim_rgb, k)["mean_ms"]
    pt_timings = {}
    for k in (500, 1000, 2000):
        pt_timings[k] = bench_cdist(rows_patch_pt, dim_xyz, k)["mean_ms"]
    fus_timings = {}
    for k in (500, 1000, 2000):
        fus_timings[k] = bench_cdist(rows_patch_pt, dim_fus, k)["mean_ms"]

    rgb_scored_ms = extrapolate_ms(rgb_timings[10000], 10000, rgb_target)
    xyz_scored_ms = extrapolate_ms(pt_timings[2000], 2000, xyz_target)
    fus_scored_ms = extrapolate_ms(fus_timings[2000], 2000, xyz_target)

    # per-sample estimate: one ViT forward per sample; scoring for the three
    # memory banks (rgb, xyz, fusion) on the test side (train side also scores
    # via late-fusion pass).
    per_sample_test_ms = dino["p50_ms"] + rgb_scored_ms + xyz_scored_ms + fus_scored_ms
    per_sample_train_ms = dino["p50_ms"] + xyz_scored_ms + fus_scored_ms
    train_count = 2656
    test_count = 1197
    coreset_ms_per_class_train = 1.5 * (rgb_scored_ms + xyz_scored_ms + fus_scored_ms)
    coreset_build_ms = coreset_ms_per_class_train * 10

    est_train_ms = per_sample_train_ms * train_count
    est_test_ms = per_sample_test_ms * test_count
    est_total_ms = est_train_ms + est_test_ms + coreset_build_ms

    result = {
        "label": "M3DM CPU runtime ESTIMATE (not a reproduction; CPU-only machine, "
                 "CUDA-only pointnet2_ops/knn_cuda not runnable)",
        "hardware": f"{platform.machine()}, cores=8, ram_audit_gb=8.42",
        "torch_threads": 8,
        "dino_vit_p8_224_per_sample_ms": dino,
        "cdist_mean_ms_measured": {
            "rgb_784x768": rgb_timings,
            "pt_3136x1152": pt_timings,
            "fus_3136x1920": fus_timings,
        },
        "expected_coreset_rows": {
            "rgb": rgb_target,
            "xyz": xyz_target,
            "fusion": xyz_target,
        },
        "scored_ms_per_sample": {
            "rgb": rgb_scored_ms,
            "xyz": xyz_scored_ms,
            "fusion": fus_scored_ms,
        },
        "per_sample_est_ms": {
            "test": per_sample_test_ms,
            "train_pass": per_sample_train_ms,
        },
        "datasets": {"train_count": train_count, "test_count": test_count},
        "estimated_total_ms": est_total_ms,
        "estimated_total_hours": est_total_ms / 3.6e6,
    }
    print(json.dumps(result, indent=2))

    fname = f"cpu_microbenchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(f"{args.out_dir}/{fname}", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("wrote", fname)


if __name__ == "__main__":
    main()