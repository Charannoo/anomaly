# PHASE3C_FROZEN — E3C synthetic-supervised dual reliability gate

Phase 3C is APPROVED and FROZEN as a diagnostic/negative-result experiment.
Later phases must not retrain it, re-tune it on test metrics, or silently
alter its supervision. It is retained as an ablation.

- Freeze date: 2026-09-18. No VCS; identity = this doc + gate checkpoint +
  split manifest + gate-data + metric tables. Phase 2 hashes re-verified.
- Artifacts: `checkpoints/E3C_gate/` (gate_best.pt 9.5 KB, feat_norm.json,
  config.json, synth_val_metrics.json), `experiments/calibration/
  E3C_split_manifest.json`, `experiments/gate_data/` (10×.npz, 66,400
  synthetic variants), `experiments/tables/E3C_*.csv`,
  `experiments/predictions/E3C_gate/`, `experiments/figures/gate/`.
- Code: `src/xmvad/models/fusion/reliability_gate.py` (1492 params),
  `src/xmvad/fusion/gate_features.py`, `scripts/{generate_gate_data,
  train_reliability_gate,evaluate_gate}.py`. Supervision: train-normal
  synthetic only (CLEAN/RGB_ONLY/DEPTH_ONLY/JOINT × mild/medium/strong).

## What was found (honest summary)

- Synthetic validation was strong: detection dominant-acc 0.908, map 0.830,
  rising with severity. Controlled behavior confirmed — not an attribution
  claim.
- Real image-level transfer FAILED: E3C I-AUROC MEAN .605 vs fixed-mean
  .640 and best-E3B .671; fixed/broken counts adverse (e.g. bagel 1/50).
- Localization transfer was BETTER: P-AUROC +.012 / AUPRO +.024 over
  best-single (potato .881, peach .790 — best in their rows); below fixed
  mean (.915/.722).
- The gate became RGB-biased on real data (w_rgb p50 at 0.95 ceiling in
  most categories; rope correctly depth-dominant). Synthetic RGB
  perturbations trigger RGB-branch responses far more cleanly than real
  defects do — max-score evidence does not transfer. Confidence floor
  (alpha=0.1) was too weak to matter under saturated softmax.
- Defects: carrot/hole .907 and foam/color .888 recover toward RGB;
  rope/open .665 and bagel/crack .835 lose vs fixed fusion;
  potato/cookie-contamination stay broken.

## Lesson carried forward

Detection and localization reliability need separate treatment (dual heads
confirmed directional), and synthetic *detection* supervision needs
realism work (subtler, edge-concentrated anomalies; top-k rather than max
evidence). E4 isolates feature interaction WITHOUT any reliability gate.
