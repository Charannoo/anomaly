# PHASE 5 — Paper-protocol hardening + modality-aware explainability (complete)

Status: **COMPLETE and FROZEN.** Phase 5 adds official-protocol metrics at
native GT resolution and a safety-first, controlled modality-attribution
benchmark around the frozen E4 fusion. Full report: the A-O summary in the
session transcript. No test-set tuning; no causal-attribution claims.

## 1. Native-GT protocol hardening

- `scripts/eval_e4_native_resolution.py` re-evaluated E4 with maps resized to
  each category's native GT resolution (bilinear, no thresholding) using the
  official MVTec AUPRO (8-connectivity, FPR lim 0.3). Memory-bounded streaming
  variant `pro_curve_stream`/`aupro_stream` added to `src/xmvad/metrics/pro.py`
  (identical to `pro_curve` to <1e-12).
- Outputs: `experiments/tables/E4_concat_metrics_native.csv`,
  `experiments/tables/E4_official_metric_verification.csv`.
- I-AUROC unchanged by construction (image-level, 256); P-AUROC and AUPRO rise
  slightly at native resolution (only tire AUPRO -0.0010).

## 2. Baseline provenance

- `docs/BASE_PAPER_2022.md`: exact MVTec 3D-AD numbers (arXiv:2112.09045),
  M3DM (CVPR 2023) comparator, and the representation-fairness warning.
- `scripts/build_paper_baseline_table.py`:
  `experiments/tables/paper_baseline_table.csv` +
  `docs/TABLES_PAPER_BASELINE.md`. E1/E2/E3A/E4/E7 rows are "Measured by us"
  (params, checkpoint size, CPU latency); MVTec-2022 + M3DM are "Reported by
  prior paper".

## 3. Modality-aware explainability (relative evidence, never causal)

- `src/xmvad/metrics/attribution.py`: region evidence E_rgb/E_depth (topk or
  mean on CALIBRATED maps), p_rgb = pos(E_rgb)/(pos(E_rgb)+pos(E_depth)+eps),
  classes RGB/DEPTH-DOMINANT / JOINT / UNCERTAIN, decision ablation, gamma
  intervention helpers, AttributionConsistency, confusion metrics, spatial
  metrics. Zero-evidence and non-finite handling are unit-tested.
- `scripts/generate_attribution_data.py`: deterministic controlled benchmark
  (TRAIN-NORMAL only; RGB_ONLY / DEPTH_ONLY / JOINT x mild/medium/strong,
  10 categories x 360 samples; seed = md5 of claim). Data:
  `experiments/attribution/attribution_data/<cat>.npz`.
- `scripts/eval_attribution.py` -> `experiments/tables/attribution_metrics.md`:
  pooled dominant-modality accuracy 0.781, macro-F1 0.873,
  AttributionConsistency 0.848, fused spatial pixel-AUROC ~0.96; confusion
  blocks show strong RGB attribution (0.928) and weaker depth (0.656).
- `scripts/intervene_gamma.py`: zero-gate tests of the fusion residual
  (gamma_rgb=0 / gamma_depth=0 / both=0, no retraining). Suppressing a gate
  generally INCREASES scores, i.e. the residual aligns fused features with the
  teacher (discrepancy-favourable regulariser), not an amplifying shortcut.
- `scripts/analyze_real_evidence.py` + 6 operator figures
  (`experiments/figures/explainability_failures/`): real evidence on 1,197
  defects inside GT (35.4% RGB-dominant, 4.9% depth-dominant, 38.9% joint,
  20.8% uncertain); rope/open is strongly depth-dominant; failures and
  strongest RGB/depth candidates exported as CSV.

## 4. Deliverables

README/title/identity updated: **E4 = "XMV-AD Residual Fusion"** (final
model), E7 = "XMV-Attention Ablation". `docs/ARCHITECTURE.md` and
`docs/RESEARCH_PLAN.md` updated; historical phase docs kept intact for
provenance.

Full test suite: **94 passed, 1 skipped** (Phase 5 adds
`tests/test_attribution.py`; `aupro_stream` equivalence test in
`tests/test_metrics.py`).

## 5. Out of scope (unchanged)

No INT8 / ONNX / OpenVINO / dashboard; no new architecture; no test-set
tuning; M3DM is reported as a prior-paper comparator only, never reproduced.
No VCS.

## 6. Honest limitations

- Evidence is RELATIVE anomaly evidence inside a region, not causal.
- Decision deltas ({Delta_RGB, Delta_DEPTH}) measure removal sensitivity of a
  feature-level fusion; they are not perfectly additive or interventional
  claims (see attribution.py docstring).
- Depth attribution is weaker than RGB on real and synthetic data (reprojected
  Z-depth specular/registration noise) - disclosed, not hidden.
- The gamma residual suppresses student-teacher discrepancy (negative Deltas);
  this means E4's fusion acts as a regulariser, so "residual contribution"
  should be read in that direction.