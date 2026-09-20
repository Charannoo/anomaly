# Experiments (protocol)

## Order

E0 sanity (stats/visuals, no claim) -> E1 RGB-only -> E2 depth-only ->
E3 late fusion -> E4 concat -> E5 one-way attention -> E6/E7 bidirectional XMV ->
E8 heads ablation -> E10-12 dim {64,96,128} -> E13/14 FP32/INT8.

## Fairness

Same splits, resolution, masks, preprocessing, seed policy, metric code.
Seeds {42, 123, 2026} for final runs; mean +/- std where meaningful.
Every run saves config.yaml + metrics.json + environment.json + checkpoint +
log + git hash under `experiments/runs/<timestamp>_<experiment>/`.

## Tables (all TBD in Phase 1)

- ablation.csv: experiment, category, seed, I/P-AUROC, AUPRO, AUPRC, params,
  model_mb, cpu p50/p95, RAM, fps, attribution acc/macro-F1.
- Per-category I-AUROC/P-AUROC/AUPRO incl. failure analysis.
- Robustness: depth missing/noise, RGB lighting/blur, alignment shifts.

## Reproduce

`bash scripts/reproduce_all.sh` after dataset + weights are present.
