# H0 — Official M3DM Reproduction Report on MVTec-3D AD

> **Phase H0 of the XMV-AD-H (high-accuracy) research track.**
> Scope: environment audit, frozen-track hash, M3DM pin + code understanding,
> dataset/preprocessing verification, reproduction attempt. This report answers
> the roadmap's A–O questions honestly. **No number is fabricated.**
> Super-rule 1 (never fabricate) is binding: where the official pipeline could
> not be executed, we say so and give an estimate, never a reproduction number.

---

## A. Machine hardware (our machine)

Measured 2026-09-18T16:51:21Z (`experiments/high_accuracy/environment.json`):

| Item | Value |
|---|---:|
| OS | Windows 11 (win32, AMD64) |
| Hostname | Hiran |
| CPU | 4 physical / 8 logical cores (Intel) |
| RAM total | 8.42 GB |
| RAM available (at audit) | 0.91 GB |
| GPU | **None** (`torch.cuda.is_available() == False`) |

M3DM's own hardware (paper App. A): experiments on a single Nvidia Tesla V100
and up to 50 GB memory consumption — not stated to be GPU VRAM. We have no CUDA
device and far less RAM.

## B. Software environment (our machine)

| Package | Version |
|---|---:|
| Python | 3.13.6 |
| torch | 2.14.0+cpu |
| numpy | 2.4.6 |
| onnxruntime | 1.30.0 |
| openvino | 2026.4.0 |
| nncf | 3.4.0 |
| torchvision | 0.29.0+cpu |
| scipy / sklearn | 1.18.1 / 1.9.1 |
| matplotlib | 3.11.1 |
| timm (isolated env only) | 1.0.29 |
| git | 2.54.0.windows.1 |
| CUDA toolkit | not present |

Official M3DM env (README): Python 3.8, PyTorch 1.9.0, CUDA 11.3, plus the
compiled deps `KNN_CUDA-0.2` and `pointnet2_ops_lib` (Pointnet2_PyTorch).

## C. Official M3DM repo + commit pinned

- Upstream: `TencentYoutuResearch/AnomalyDetection-M3DM` (paper GitHub
  `nomewang/M3DM`), cloned read-only to `third_party/m3dm/`.
- Pinned HEAD: **`bd8151a4563ee7445bdc32d01b2a3b384e4d54f5`** ("The initial
  code", 2023-04-10; matches upstream HEAD verified by `git ls-remote`).
- Source files audited: `main.py`, `m3dm_runner.py`, `dataset.py`,
  `feature_extractors/{features,multiple_features}.py`,
  `models/{models,feature_fusion}.py`, `utils/{mvtec3d_util,au_pro_util,utils,preprocessing}.py`.
- Scoring/metrics internals documented (see section M).
- The repo ships **no pretrained checkpoints** (no `checkpoints/` dir): Point-MAE
  weights, DINO weights, and the UFF fusion weights must be obtained/trained per
  README.

## D. Pretrained checkpoints used by the official pipeline

Per M3DM README + code (`models/models.py`, `feature_fusion.py`) the full
DINO+Point_MAE+Fusion setting requires:

| Component | Source | Path in repo |
|---|---|---|
| DINO ViT-B/8 | timm `vit_base_patch8_224_dino` (DINO offline-url variant) | downloaded by timm |
| Point Transformer (Point-MAE) | Point-MAE `pretrain.pth` | `checkpoints/pointmae_pretrain.pth` |
| UFF fusion module | trained with `fusion_pretrain.py` (750 steps guideline) | `checkpoints/checkpoint-0.pth` |

None of these are in the repo; the UFF module is trained by the operator. We
therefore cannot pin byte-identical weights without downloading Point-MAE/DINO
(possible) and then training UFF ourselves (requires feature extraction first).

## E. Preprocessing: official vs our dataset copy

The M3DM **paper**'s stated preprocessing ("Data Preprocess", §4.1): estimate the
background plane with RANSAC, remove points within 0.005 distance, **set the
corresponding pixel of removed points in the RGB image to 0**, then **resize both
the position tensor and the RGB image to 224x224** (matching the feature extractor
input size).

The official **code** (`utils/preprocessing.py`) implements: RANSAC plane removal
(plane fitted on the 10-px edges, pts within `distance_threshold=0.005` zeroed in
both XYZ and RGB) → square 100-multiple zero-padding (no-op for our 800x800) →
DBSCAN connected-component cleaning (outliers zeroed in XYZ and RGB). Important
faithful-reproduction detail: the code computes the zeroed RGB **in memory** but
only persists the `xyz/*.tiff` (the `rgb` and `gt` saves are commented out).
Consequently, after official preprocessing the on-disk RGB is unchanged, and
`dataset.py` reads RGB directly from the png; 224x224 resizing happens at **load
time** (RGB bicubic, position tensor nearest). We therefore reproduce exactly what
the published numbers used: zeroed XYZ on disk, unmasked RGB, 224x224 at load.

- Our raw frozen copy `data/raw/mvtec3d` is **unpreprocessed** (raw organized
  point clouds, 800x800, zero = invalid) and will not be modified.
- For H0-R a separate copy is created at `data/m3dm_repro/mvtec3d/` and the
  official preprocessing runs **only on that copy** (see
  `docs/H0-R_M3DM_REPRODUCTION.md`, sections D/F).
- Test-count discrepancy: paper App. A says **1137 test** samples; our
  official-layout copy measures **1197 test**. We do not delete samples. The
  loader-level resolution is performed in H0-R (section D of the H0-R report).
- **H0-R resolution (MEASURED 2026-09-18):** M3DM's pinned `TestDataset`
  enumerates all `test/` subdirs (`os.listdir`) and treats every non-`good`
  folder as anomaly (label 1, gt paired). The on-disk MVTec-3D-AD v1.0 layout
  contains, per category, per-defect folders **plus** a `test/combined/` folder
  of *multi-defect* samples whose `gt` encodes defect type per `class_ids.json`
  (e.g. bagel: pixel 255=contamination, 254=crack, 253=hole). We verified `combined/`
  files are content-disjoint from per-defect folders (hashes), so `os.listdir`
  produces **no double-counting**. Across all 10 categories the loader's
  as-loaded totals match official MVTec-3D-AD Table 1 exactly
  (train 2656, val 294, test good 249 + test anomalous 948 = 1197). The paper's
  "1137" is not explainable by the official split and is treated as REPORTED BY
  PRIOR WORK; no dataset samples are deleted.

## F. Paper-reported M3DM metrics (primary source, REPORTED BY PRIOR WORK)

Source: arXiv:2303.00601v2 (CVPR 2023), full setting (RGB+3D, "Ours").
Full table in `experiments/high_accuracy/tables/H0_m3dm_paper_metrics.md`:

| Category | I-AUROC | P-AUROC | AUPRO |
|---|---:|---:|---:|
| bagel | 0.994 | 0.995 | 0.970 |
| cable_gland | 0.909 | 0.993 | 0.971 |
| carrot | 0.972 | 0.997 | 0.979 |
| cookie | 0.976 | 0.985 | 0.950 |
| dowel | 0.960 | 0.985 | 0.941 |
| foam | 0.942 | 0.984 | 0.932 |
| peach | 0.973 | 0.996 | 0.977 |
| potato | 0.899 | 0.994 | 0.971 |
| rope | 0.972 | 0.997 | 0.971 |
| tire | 0.850 | 0.996 | 0.975 |
| **Mean** | **0.945** | **0.992** | **0.964** |

Our computed means match the paper's stated means exactly → extraction sanity
confirmed.

## G. Our per-category metrics (MEASURED BY US)

**Not measurable on this machine.** The official pipeline imports CUDA-only
compiled ops (`pointnet2_ops`, `knn_cuda`) at module scope and calls
`pointnet2_utils.furthest_point_sample` (CUDA kernel) in `fps()`. There is no
CPU path. `H0_m3dm_reproduction.csv` records all measured columns as `NA
(measurement blocked: no CUDA GPU)` — no fabricated numbers.

## H. Reproduced mean

**Not available.** No official-pipeline run completed.

## I. Absolute delta vs paper

**Not available** (no measured values). When reproducible on CUDA hardware, this
represents the honest gap; per roadmap the reported-vs-measured split remains
mandatory.

## J. Metric-verification alignment

- **I-AUROC / P-AUROC**: sklearn `roc_auc_score` — identical definition to
  M3DM (`features.py`).
- **AUPRO**: M3DM ships its own `utils/au_pro_util.py` (integration limit 0.3,
  100 thresholds), built upon the official MVTec 3D-AD evaluation code. We do
  **not** claim numerical identity with our evaluator based on source inspection
  alone. For the H0-R reproduction, the **primary** AUPRO must come from M3DM's
  own evaluator; a formal cross-check against our evaluator on identical maps
  (H0-R.9) is the proof of equivalence, and no equivalence claim is made until
  that check exists.
- **Resolution**: M3DM evaluates pixel metrics at **224x224** (gt mask resized
  NEAREST to 224 in `dataset.py`; s_map upsampled to 224 and blurred). Our frozen
  pipeline evaluates at different resolutions — for a fair delta, any future run
  must reuse M3DM's 224x224 protocol.

## K. Runtime

No reproduction ran. CPU **estimate** (not a reproduction) from
`benchmark_m3dm_cpu_estimate.py`
(`experiments/high_accuracy/runs/cpu_microbenchmark_<ts>.json`):
- DINO ViT-B/8 forward on this CPU: ~1.07 s p50 / sample.
- Full-estimate total for 2656 train + 1197 test: **~12.8 h** (extrapolated
  cdist; excludes CUDA-only Point-Transformer/grouping cost entirely and exceeds
  available RAM, so the real CPU runtime would be far higher and likely
  memory-infeasible). Official environment: V100-class single GPU, "up to 50 GB
  memory" (see L — not necessarily GPU VRAM).

## L. GPU / RAM usage

- Reported by paper (App. A): "All experiments are run on a single Nvidia Tesla
  V100 and cost at most 50 GB of memories for the full setting." — M3DM reports
  experiments on one Tesla V100 and up to 50 GB memory consumption. This is
  **not stated to be GPU VRAM**.
- Our machine: no GPU, 8.42 GB RAM → full-setting reproduction is **infeasible**
  in this hardware class even ignoring the CUDA software dependency.

## M. Problems / findings

1. **CUDA hard dependency (blocking).** `pointnet2_ops` + `knn_cuda` import at
   module scope; `fps()`/`Group`/DLF depend on CUDA kernels. Cannot run on CPU.
2. **No released checkpoints.** Fusion (UFF) weights must be trained; DINO/Point-MAE
   weights must be downloaded. Byte-pinned reproduction of the paper's exact model
   is therefore only partial without training UFF ourselves.
3. **Mean/std bugs in official coreset code.** In `TripleFeatures.run_coreset`
   the normalization stats are cross-wired (`xyz_std = std(rgb)`, `rgb_mean =
   mean(xyz)`, `fusion_mean = mean(xyz)`, `fusion_std = std(rgb)`); same pattern
   in `DoubleRGBPointFeatures`. Faithful reproduction reproduces these; any
   "improvement" must be documented as a deviation.
4. **Test-count discrepancy (1137 vs 1197) — RESOLVED in H0-R.** M3DM's pinned
   loader iterates all `test/` subdirs including `test/combined/`; we verified the
   combined samples are disjoint multi-defect annotations, and that as-loaded the
   per-category counts match official MVTec-3D-AD Table 1 exactly (test = 249
   good + 948 anomalous = 1197). See section E above.
5. **UFF training dependency.** Official UFF trains on all MVTec-3D train features
   (`fusion_pretrain.py`, 750-step guideline) before the main run — a hidden
   training stage, not a pure inference reproduction.
6. **Nondeterminism.** coreset `SparseRandomProjection(random_state=None)` by
   default (args.random_state None) → run-to-run variation; seeds only set for
   torch/numpy/random, not sklearn.

## N. Is the H0 outcome trustworthy?

The **negative result is trustworthy**: we can substantiate that the official M3DM
full pipeline cannot run on this machine (verified imports + CUDA kernel calls +
no GPU). The **paper numbers in section F are trustworthy**: extracted from the
primary source and cross-checked against the paper's own stated means. There are
**no measured numbers** to be trusted or distrusted — honesty preserved.

## O. Recommendation

**FIX REPRODUCTION FIRST — do not start H0B/H1 on this hardware.**

Blocking conditions found in H0:
- Full M3DM requires a CUDA GPU (pointnet2_ops/knn_cuda); this machine is
  CPU-only with 8.42 GB RAM and cannot run it.
- A CPU re-implementation would change the method (violates reproduction rules).

Recommended next steps (require review / a CUDA machine or WSL2+GPU):
1. Acquire or access a CUDA/Linux GPU node.
2. Run official `utils/preprocessing.py` on a fresh MVTec-3D AD copy.
3. Download DINO + Point-MAE weights; train UFF per README (750 steps).
4. Run `main.py --method_name DINO+Point_MAE+Fusion --use_uff --memory_bank
   multiple` with paper defaults (M=1024, S=128, f_coreset=0.1, eps=0.9,
   img_size=224, lambdas per README) on all 10 categories.
5. Record per-category I-AUROC/P-AUROC/AUPRO and compare vs section F, then
   decide PROCEED (H0B/H1) vs further FIX.

---

*Generated 2026-09-18. Hash manifest of the frozen XMV-AD Lite track:
`experiments/high_accuracy/frozen_artifact_manifest.csv`.*