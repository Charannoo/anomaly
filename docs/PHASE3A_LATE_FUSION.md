# PHASE3A_FROZEN — E3 late RGB+Depth anomaly fusion

Phase 3A is APPROVED and FROZEN. Later phases may read its predictions and
tables but must not silently alter its method, calibration, or results.

- Freeze date: 2026-09-18. No VCS; identity = this doc + calibration JSON +
  metric tables + Phase 2 checkpoint hashes (re-verified identical).
- Artifacts: `experiments/calibration/E3_late_fusion_calibration.json`,
  `experiments/tables/E3_{late_fusion_metrics,vs_baselines,bootstrap_ci,defect_level,complementarity}.csv`,
  `experiments/predictions/E3_late_fusion/`, `experiments/figures/late_fusion/`.
- Scripts: `scripts/calibrate_late_fusion.py`, `scripts/run_late_fusion.py`.
  No retraining occurred; no test data in calibration.

## Method

- Validation-normal robust calibration per category×modality:
  `Z = (A − median)/MAD·1.4826` (std→1.0 fallbacks), depth pools valid
  pixels only, no clamping. Same method everywhere.
- Declared E3 baseline: fixed calibrated 50/50 mean fusion; depth-invalid
  pixels use RGB evidence only. Secondary (reported separately):
  calibrated max fusion.
- Image scores: max (primary, matches official code) + top-0.1%/top-1% means.

## Full result (test, calibrated mean fusion, max readout)

| category | RGB I/P/AUPRO | Depth I/P/AUPRO | Fusion I/P/AUPRO | ΔBestSingle I/P/AUPRO |
|---|---|---|---|---|
| bagel | .733/.958/.840 | .720/.781/.521 | .873/.978/.897 | +.140/+.020/+.057 |
| cable_gland | .831/.845/.534 | .166/.788/.487 | .802/.806/.409 | −.029/−.038/−.126 |
| carrot | .804/.940/.822 | .546/.789/.561 | .696/.939/.820 | −.108/−.001/−.002 |
| cookie | .465/.841/.462 | .494/.792/.642 | .513/.927/.728 | +.019/+.086/+.086 |
| dowel | .568/.869/.575 | .538/.814/.499 | .638/.891/.625 | +.069/+.022/+.050 |
| foam | .790/.893/.705 | .481/.451/.100 | .744/.839/.587 | −.046/−.055/−.119 |
| peach | .385/.937/.781 | .556/.722/.393 | .422/.937/.774 | −.134/+.000/−.007 |
| potato | .415/.969/.886 | .673/.720/.457 | .463/.948/.820 | −.210/−.021/−.067 |
| rope | .813/.784/.530 | .662/.857/.775 | .690/.960/.845 | −.123/+.103/+.070 |
| tire | .597/.914/.719 | .505/.828/.482 | .559/.923/.720 | −.038/+.008/+.001 |
| MEAN | .640/.895/.685 | .534/.754/.492 | .640/.915/.722 | −.000/+.020/+.037 |

Max-fusion secondary MEAN: I .60 / P .90 / AUPRO .70 (worse than mean).

## Negative transfer cases (fixed 50/50 destroys branch wins)

potato/contamination (depth .719→.352), carrot/hole (RGB .969→.862),
foam/color (RGB .913→.823), cable_gland image level (RGB .831→.802 with
depth anti-correlated). Fusion wins: bagel/crack (.837→.953),
bagel/combined (.808→.921), rope/open (.816→.842).

## Bootstrap (1000 resamples, image AUROC max)

Significant diffs only: bagel fused>both; carrot fused<RGB; potato
fused<depth. All other fusion-vs-branch Δ CIs cross 0 — most image-level
effects within noise at this readout.

## Limitations

1. Fixed weights cannot protect a strong branch from a weak one.
2. 256px eval resolution (inherited). 3. Single seed. 4. Complementarity
   classes use validation-p95 thresholds (qualitative only).
