# H0-R: M3DM Feature Extraction Pipeline (verified against official repo)

Pinned commit: `bd8151a4563ee7445bdc32d01b2a3b384e4d54f5`
Environment: WSL2 Debian, Python 3.8.20, torch 1.9.0+cu111, 3 GB guest RAM
on an 8 GB Windows host (guest over-commit handled via disk swap, guarded).

## 1. Data (MVTec-3D-AD, official layout)

Per category, three folders: `train`, `test`, `validation`.

| split | loader samples | note |
|---|---:|---|
| train | 2656 | all `good`, RGB 224×224 `png` + organized XYZ `.tiff` per sample |
| test | 1197 | all defect dirs **plus** `test/good` (loader reads every dir under test) |
| validation | 294 | present, not consumed by the official loader |

The official paper text cites 1137 test samples; the loader here enumerates
1197. This is reported honestly; no sample counts were added/removed. Details
in `m3dm_loaded_sample_manifest.csv` (per-sample: set, sample id, defect type).

Each organized point cloud `.tiff` is loaded and converted to an unorganized
point cloud (`organized_pc_to_unorganized_pc`). Zero points are dropped once
(`nonzero_indices`); that mask point count is **not** rescaled back to the
224×224 grid, matching official behavior.

## 2. Feature extractors

### 2.1 RGB (DINO ViT-B/8)
- Backbone: `vit_base_patch8_224_dino` from the `timm` hub, `pretrained=True`.
  Runtime weights: `/root/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth`
  (SHA256 `575f0efc…`, 343279349 B). The repo-local
  `checkpoints/dino_vitb8_state_dict.pt` is **not** loaded by this runner
  (SHA256 `cbff1f0b…`, 343284245 B; documented, unused).
- Image: 224×224 RGB → patch size 8 → 28×28 grid, 768-d tokens.
- Memory bank patch shape: `[191296, 768]` for bagel (784 patches × 244 samples)
  i.e. `torch.nn.functional.unfold` style per-sample 784 patches.

### 2.2 XYZ (Point-MAE)
- Backbone: Point_MAE (`pointmae_pretrain.pth`, SHA256 `27ded93…`, 348288755 B).
- Input: unorganized point cloud of the sample, `group_size=128`,
  `num_group=1024` → 1024 groups × 128 points. Group centers + neighbour
  indices used to interpolate back to original point indices.
- XYZ features then padded back onto the 224×224 grid, passed through
  `AvgPool2d(3, stride=1)` then `AdaptiveAvgPool2d((56,56))` → 56×56 = 3136
  patches per sample; feature dim 1152 per patch.
- Bagel memory bank: `[765184, 1152]` (= 3136 × 244).

### 2.3 Feature dims (used by fusion blocks)
- XYZ: 1152-d; RGB: 768-d.
- `FeatureFusionBlock(1152, 768, mlp_ratio=4.)` (models/feature_fusion.py),
  initialized from `checkpoints/checkpoint-0.pth` when `--use_uff`.

## 3. Stage-1 (no UFF): DINO+Point_MAE, memory bank = multiple

For each class separately (this repro runs one class per process to stay under
the 3 GB cap):

1. `fit`: extract train × samples → `patch_xyz_lib`, `patch_rgb_lib`
   (memory banks `multiple` = separate xyz and rgb banks, official default).
2. `coreset` per bank, **official** settings `f_coreset=0.1`, `eps=0.90`:
   - `SparseRandomProjection(eps=0.9)` → projected bank.
   - Greedy `min(dist)` selection of 10% indices.
   - Counts: bagel → xyz 765184→coreset, rgb 191296→coreset (see log).
3. Late fusion: per test sample, `torch.cdist(patch, coreset_bank)` min
   distances → `s` (reweighted) and `s_map` (interpolated 224×224, blurred).
   Official `/1000` reweighting preserved verbatim.
4. `SGDOneClassSVM(random_state=42, nu=0.5, max_iter=1000)` fitters on the
   concatenated `s`/`s_map` lib (official `linear_model`).
5. Predict → image labels from `fuser.decision_function`, pixel maps from
   `seg_fuser`. Metrics: image/image-pixel ROC-AUC + AU-PRO by the official
   `calculate_au_pro`.

### Memory-lean equivalence patches (not scoring changes)
All reproducibility patches are pure memory/CPU-shape adapters; every float
produced for the reported numbers follows the official arithmetic:

| patch | file | reason | math impact |
|---|---|---|---|
| coreset projection+chunked transform | `features.get_coreset_idx_randomp` | full fp64 projection OOM under 3 GB | identical selector; fp64→fp16 rounding only |
| late-fusion chunked `cdist`/argmin | `features.add_sample_to_late_fusion_mem_bank`, `compute_s_s_map` | full cross cdist OOM | per-row cdist identical |
| `s_star` reweight keep `/1000` | `compute_single_s_s_map` | official scaling | preserved exactly |
| cross-wired normalize quirk | `run_coreset` | official bug | reproduced, **not** fixed |

Deterministic coreset disk cache: `/opt/tmp/coreset_cache/coreset_<sha256>.pt`,
keyed on lib shape + lib head/tail bytes + fp16 column sums + `(n, eps, random_state)`
so any change in the shuffle=False extraction invalidates the cache. Bagel run
logged `CORESET CACHE HIT` (2/2 banks previously computed).

## 4. UFF (fusion training) — to be run after all stage-1 classes
- `main.py --method_name DINO+Point_MAE --memory_bank multiple --save_feature True`
  writes per-sample `cat(xyz_patch_full_resized, rgb_patch_resize)` (3136 ×
  (1152+768) = 3136×1920) into `datasets/patch_lib/`.
- `fusion_pretrain.py` trains `FeatureFusionBlock(1152,768)` with
  contrastive head; sample = stack of patch tensors; split at `1152`.
- Optimizer AdamW `(0.9,0.95)`, `blr=0.002`, 10 epochs, `batch_size=64`,
  `accum_iter=16` (README). Checkpoints `checkpoint-{epoch}.pth`.
- In-model split: `samples[:,:,:1152]` (xyz), `samples[:,:,1152:]` (rgb);
  `FeatureFusionBlock` fuses and the contrastive loss masks zero-padded rows
  (`patch_no_zeros_indices` on xyz).

## 5. Full M3DM (DINO+Point_MAE+Fusion) — to be run after UFF
- `main.py --method_name DINO+Point_MAE+Fusion --use_uff
  --fusion_module_path checkpoints/checkpoint-0.pth`
- Three memory banks (xyz, rgb, fusion) at f_coreset=0.1 / eps=0.9 inside
  `TripleFeatures`, three sets of `s`/`s_map` + detection & segmentation
  fusers. Report objective is the mean of 10 categories vs paper
  means 0.945 (I-AUROC) / 0.992 (P-AUROC) / 0.964 (AUPRO).

## 6. Acceptance
- Mean delta ≤ 0.01 strong; 0.01–0.02 investigate; > 0.02 untrustworthy.
- Category-level review (bagel already measured: 0.988 / 0.994 / 0.966 vs
  paper 0.994 / 0.995 / 0.970).