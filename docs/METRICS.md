# Metrics

## Shared with M3DM (exact MVTec 3D-AD protocol required)

1. Image-level AUROC — implemented (`metrics/detection.py`), unit-tested.
2. Pixel-level AUROC — implemented (`metrics/localization.py`), unit-tested.
3. PRO / AUPRO — implemented (`metrics/pro.py`) as the official MVTec
   3D-AD algorithm (evaluation_code v2.0): 8-connectivity, exact curve over
   all unique scores, trapezoid to FPR 0.3 / 0.3. VERIFIED: standalone
   cross-check of official script vs our module agrees to <1e-9 on synthetic
   fixtures and real 256px prediction maps; golden values pinned in
   `tests/test_metrics.py::test_aupro_matches_official_golden`.

Evaluation resolution in Phase 2: config `input_size` (256×256); maps and GT
compared at that resolution (documented choice, revisit in Phase 6).

## Additional (Phase 6)

Pixel/Image AUPRC, IoU at calibrated threshold, Dice/F1 localization.

## Efficiency (Phase 7)

Params, file size, peak RAM, CPU p50/p95 latency, FPS, MACs/FLOPs if reliable.
Protocol: batch 1, warmup 50, >=200 iters, sizes 256/320/384, controlled threads.

## Explainability (Phase 5/6)

Attribution accuracy/macro-F1 + confusion matrix on synthetic RGB/depth/joint
groups; ablation agreement; score drop; optional deletion/insertion curves.

All values TBD — no numbers are claimed in Phase 1.
