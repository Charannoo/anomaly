# Explainability

## Operator output

Deterministic categories: RGB_DOMINANT / DEPTH_DOMINANT / JOINT / UNCERTAIN,
from calibrated score contributions + branch ablation:

S_full vs S_no_rgb vs S_no_depth; Delta_RGB = S_full - S_no_rgb, etc.
Thresholds calibrated on held-out validation. No fake-precision percentages
until calibration justifies them.

## Attention is not explanation

Cross-attention maps are "interaction visualizations", never causal proof.
Faithfulness requires modality ablation, perturbation, Integrated Gradients
(captum), anomaly-map evidence, deletion/insertion where appropriate.

## Controlled benchmark (Phase 5)

Deterministic synthetic groups (see `src/xmvad/data/synthetic_anomalies.py`):
A) RGB-only (color/stain/brightness/scratch), B) depth-only (dent/bump/hole),
C) joint. Attribution accuracy/macro-F1 + confusion + ablation agreement.
Synthetic data evaluates attribution only — never replaces the real benchmark
or trains the detector.

Phase 1: perturbation utilities exist; attribution heads/metrics land in
Phases 5-6.
