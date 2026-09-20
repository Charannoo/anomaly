# PHASE5_FROZEN — Phase 5 (explainability) freeze record

Status: **FROZEN.** Phase 5 approved and locked before Phase 6 deployment
work. Phase 5 inputs to Phase 6 are the frozen E4 checkpoints, the official
native-resolution metric protocol, and the (unmodified) preprocessing.

## What is frozen

- All E1–E7 checkpoints (final E4 checkpoints at
  `checkpoints/E4_concat_fusion/<cat>_best.pt`).
- Phase 5 native-resolution tables:
  `experiments/tables/E4_concat_metrics_native.csv`,
  `experiments/tables/E4_official_metric_verification.csv`,
  `experiments/tables/paper_baseline_table.csv`.
- Controlled attribution benchmark + metrics
  (`experiments/attribution/`, `experiments/tables/attribution_metrics*.csv`,
  `attribution_metrics.md`), decision ablation + gamma intervention tables,
  real-defect evidence tables, and operator explanation figures
  (`experiments/figures/explainability_failures/`).
- README / model identity: **E4 = XMV-AD Residual Fusion** (final model),
  E7 = XMV-Attention Ablation.

## Freeze checks performed (Phase 6 milestone 0)

1. **E4 checkpoint hashes** — SHA-256 of all 10 final E4 checkpoints recorded
   in `experiments/artifacts/e4_checkpoint_hashes.csv` (see below).
2. **Phase 5 metric tables exist** — verified (list below).
3. **pytest** — full suite green (see below).
4. **Environment information** — stored in
   `experiments/artifacts/environment.json` and `requirements.txt`.

## Checkpoint hashes

SHA-256 computed over `checkpoints/E4_concat_fusion/<cat>_best.pt`
(see `experiments/artifacts/e4_checkpoint_hashes.csv`). Hashes are recorded
for integrity + identity of the frozen deployment artifacts; they are not a
cryptographic seal and the command used is recorded for reproduction.

## Phase 5 tables verified present

E4_concat_metrics_native.csv, E4_official_metric_verification.csv,
paper_baseline_table.csv, E4_concat_calibration.json, attribution_metrics.csv,
attribution_metrics_detail.csv, attribution_metrics.md,
gamma_intervention_summary/detail/pooled.csv, gamma_intervention_metrics.md,
real_evidence_metrics.csv, real_evidence_summary.csv, real_evidence.md.

## pytest

`python -m pytest -q` -> 94 passed, 1 skipped (recorded at freeze time).

## Environment snapshot

`experiments/artifacts/environment.json` captures Python / OS / PyTorch /
ONNX / onnxruntime / OpenVINO / NNCF versions used in Phase 6.

## Correction note (Phase 6 milestone 1, reporting audit)

The Phase 5 attribution report initially stated macro-F1 = 0.918. The Phase 6
reporting audit (`scripts/audit_attribution_metrics.py`,
`experiments/tables/attribution_metric_audit.md`) reproduced that number from
the frozen `eval_attribution.py` and found the JOINT label was always scored
correct (precision/recall forced to 1.0), inflating the average. The audit
corrected the metric computation (correct prediction -> injected kind,
otherwise MISS) and regenerated `experiments/tables/attribution_metrics*`:

- dominant-modality accuracy: **0.781** (unchanged).
- macro-F1 (3 injected modalities): **0.873** (was 0.918).
- AttributionConsistency: **0.848** (unchanged).

This is a metric-computation correction only; no model weights, evaluation
thresholds, preprocessing, or test-anomaly handling were changed.