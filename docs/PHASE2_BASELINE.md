# PHASE2_FROZEN_BASELINE

Phase 2 (E1 RGB-only + E2 depth-only teacher/student baselines) is APPROVED
and FROZEN. No Phase 2 file listed here may be silently modified by later
phases. E3+ operate on saved predictions only.

- Freeze date: 2026-09-17. VCS: none (no git repo) — freeze identity is this
  document + `experiments/tables/phase2_freeze_manifest.json` (sha256 of all
  20 checkpoints) + per-run `config.yaml` / `environment.json` / `metrics.json`.
  Re-verified post-Phase-3A: all 20 hashes identical, Phase 2 untouched.
- Configs: `configs/baseline_rgb.yaml` (experiment E1_baseline_rgb),
  `configs/baseline_depth.yaml` (E2_baseline_depth), both inheriting
  `configs/default.yaml`. Hyperparameters identical for all 10 categories
  (no per-category tuning).
- Checkpoints: `checkpoints/E1_baseline_rgb/<cat>_best.pt` and
  `checkpoints/E2_baseline_depth/<cat>_best.pt` (10 + 10, ~4.1 MB each).

## Final architecture (hardened)

- Teacher: frozen ImageNet MobileNetV3-Small; taps stage8 = features[3]
  (24 ch @ 1/8), stage16 = features[8] (48 ch @ 1/16). Depth stem: RGB
  kernels averaged over channels; no weight sharing with RGB.
- Student: compact DSConv CNN (32→48→64→96 ch) with trainable 1×1 heads to
  exactly 24/48 channels. NO trainable teacher projection — cosine
  discrepancy is computed directly against frozen teacher representations.
- Params: RGB 946,169 total / 19,161 trainable; depth 945,795 / 19,075
  (frozen teacher 927,008 / 926,720). Loss: 0.5·L8 + 0.5·L16, AdamW lr 1e-3,
  wd 1e-4, plateau scheduler, 50 epochs, batch 8, seed 42, CPU.
- Depth-mask policy: majority rule (cell valid iff ≥50% contributors valid);
  survival audit in `experiments/tables/mask_survival.csv` (stage16 keeps
  88–100% per category). Invalid cells zeroed BEFORE map upsampling; invalid
  pixels forced to 0 AFTER aggregation.

## Evaluation protocol

- Resolution 256×256 (maps + GT compared there; documented, revisit Phase 6).
- Scores: max (primary/official), plus exploratory top-0.1% / top-1% means.
- AUPRO: official MVTec evaluation_code v2.0 algorithm (8-conn, exact unique
  thresholds, trapezoid to FPR 0.3); cross-checked to <1e-9; golden test
  `test_aupro_matches_official_golden`.
- Training: normal-only (train split); validation-normal for monitoring;
  test never touched during training.

## Full results (test, max score unless noted)

| category | RGB I-AUROC | RGB P-AUROC | RGB AUPRO | Depth I-AUROC | Depth P-AUROC | Depth AUPRO |
|---|---|---|---|---|---|---|
| bagel | 0.733 | 0.958 | 0.840 | 0.720 | 0.781 | 0.521 |
| cable_gland | 0.831 | 0.845 | 0.534 | 0.166 | 0.788 | 0.487 |
| carrot | 0.804 | 0.940 | 0.822 | 0.546 | 0.789 | 0.561 |
| cookie | 0.465 | 0.841 | 0.462 | 0.494 | 0.792 | 0.642 |
| dowel | 0.568 | 0.869 | 0.575 | 0.538 | 0.814 | 0.499 |
| foam | 0.790 | 0.893 | 0.705 | 0.481 | 0.451 | 0.100 |
| peach | 0.385 | 0.937 | 0.781 | 0.556 | 0.722 | 0.393 |
| potato | 0.415 | 0.969 | 0.886 | 0.673 | 0.720 | 0.457 |
| rope | 0.813 | 0.784 | 0.530 | 0.662 | 0.857 | 0.775 |
| tire | 0.597 | 0.914 | 0.719 | 0.505 | 0.828 | 0.482 |
| MEAN | 0.640 | 0.895 | 0.685 | 0.534 | 0.754 | 0.492 |

Full tables (incl. top-k variants, AUPRC, dev latency): `experiments/tables/`.
Defect-level AUROC: `experiments/tables/defect_level_auroc.csv`.
Predictions: `experiments/predictions/{E1_baseline_rgb,E2_baseline_depth}/`.

## Known limitations

1. Image-level max readout is weak/anti-correlated on several categories
   (peach/potato RGB, cable_gland depth) despite strong localization —
   readout/calibration issue, not fixed in Phase 2 by design.
2. Depth/cable_gland systematically anti-correlated (0.14–0.21 all defects);
   depth/foam failed (AUPRO 0.10). Prime failure cases for fusion analysis.
3. 256px eval, not full-res. 4. Dataset copy is user-provided (mirror-like
   provenance); results labeled accordingly until official download verified.
4. Single seed (42); no repeats yet.
