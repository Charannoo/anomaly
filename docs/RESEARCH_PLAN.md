# Research plan (XMV-AD)

## Objective

Test whether a lightweight modality-preserving RGB-D detector with a compact
residual feature concatenation (final model) plus a cross-attention ablation
can deliver useful industrial anomaly localization + validated modality
explanations on CPU.

## Research questions

- RQ1: RGB+depth > RGB-only / depth-only?
- RQ2: bidirectional cross-attention > concat / late fusion? — **Answer: No**
  (E4 residual concat wins; E5/E6/E7 do not; see PHASE4B_ATTENTION_ABLATION.md)
- RQ3: can separate RGB/depth evidence be preserved?
- RQ4: does attribution agree with controlled RGB-only/depth-only perturbations?
- RQ5: does branch removal reduce scores consistently with claimed importance?
- RQ6: is CPU deployment practical (<100ms p95 target, <10M params goal)?
- RQ7: trade-off vs M3DM on at least one shared metric (I-AUROC/P-AUROC/AUPRO)?

## Staged plan

1. **Phase 1 (now):** data pipeline — discovery, Z-depth, masks, sync transforms,
   inspection, visualization, unit tests. No training.
2. **Phase 2:** E1 RGB-only + E2 depth-only teacher/student baselines.
3. **Phase 3:** E3 late fusion + E4 concat (+ E5 one-way ablations).
4. **Phase 4:** E6/E7 bidirectional XMV-AD + losses (distillation first).
5. **Phase 5:** attribution (ablation/IG) + synthetic perturbation benchmark.
6. **Phase 6:** full MVTec 3D-AD benchmark + M3DM comparison (per-category).
7. **Phase 7:** ONNX/OpenVINO export, INT8 PTQ, CPU benchmark, robustness.
8. **Phase 8:** operator demo + paper figures/tables.

## Integrity rules

No fabricated data/results; no test tuning; normal-only training; attention is
interaction visualization, never causal proof; per-category reporting incl.
failures; disclose RGB+Z vs RGB+XYZ gap; label every number as
literature-reported / reproduced / XMV-AD-measured.
