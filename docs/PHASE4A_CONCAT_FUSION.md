# PHASE4A_FROZEN — E4 residual feature concatenation

Phase 4A is APPROVED and FROZEN. Later phases may read its checkpoints,
predictions and tables but must not retrain or silently alter it. E4 is the
learned-fusion baseline against which cross-attention (E5/E6/E7) competes.

- Freeze date: 2026-09-18. No VCS; identity = this doc + checkpoints +
  run dirs (training curves) + predictions + calibration + tables.
  Phase 2 hashes re-verified.
- Code: `src/xmvad/models/fusion/concat.py` (ConcatFusion + E4ConcatModel),
  `scripts/train_concat_fusion.py`, `scripts/evaluate_concat.py`,
  `configs/concat_fusion.yaml`.
- Checkpoints: `checkpoints/E4_concat_fusion/<cat>_best.pt` (10, ~35 KB).

## Exact architecture

Fusion @1/16 only. Frozen Phase 2 student S_rgb/S_depth (48 ch) → 1×1 to
d=32 each → concat (64) → 1×1(64→32)+ReLU+DW3×3+BN+ReLU → dual 1×1(32→48)
deltas; S_fused = S + γ·δ, γ init 0 (identity start). Stage8 untouched.
**8738 trainable params** (only fusion module; branches frozen).
Loss: masked cosine vs frozen T16 + 1e-4 residual-magnitude penalty.
AdamW 1e-3, 50 epochs, early stopping (patience 8, val w/o penalty), seed 42,
normal-only. Final maps: E3-declared calibration refit on E4 validation maps
(`E4_concat_calibration.json`) + 50/50 mean (invalid→RGB only).

## Full metrics (test, max readout unless noted)

| cat | E4 I | E4 P | E4 AUPRO | dI vs best-single | dP | dAUPRO |
|---|---|---|---|---|---|---|
| bagel | .854 | .986 | .929 | +.121 | +.028 | +.089 |
| cable_gland | .711 | .900 | .674 | −.120 | +.055 | +.140 |
| carrot | .737 | .983 | .948 | −.067 | +.043 | +.126 |
| cookie | .571 | .965 | .853 | +.077 | +.125 | +.211 |
| dowel | .593 | .967 | .873 | +.025 | +.099 | +.298 |
| foam | .666 | .880 | .682 | −.124 | −.014 | −.023 |
| peach | .570 | .979 | .925 | +.014 | +.042 | +.144 |
| potato | .598 | .989 | .958 | −.075 | +.020 | +.072 |
| rope | .697 | .972 | .905 | −.116 | +.115 | +.130 |
| tire | .540 | .942 | .781 | −.057 | +.027 | +.062 |
| MEAN | .654/.752t | .956 | .853 | +.014/+.066t | +.061 | +.168 |

(t = top-1% variant: E4 .752 vs RGB .686.) MEAN deltas vs fixed fusion:
+.014 I / +.041 P / +.131 AUPRO. E4 beats every score fusion on localization.

## Map-change results (E4 vs Phase 2)

RGB maps change substantially (mean|Δ| 0.02–0.07, corr 0.69–0.92);
depth maps stay closer (0.01–0.03, corr 0.83–0.98). Asymmetric: RGB absorbs
more cross-modal influence. Gammas non-zero everywhere (|γ| .22–.38, mixed
signs per category) — no residual collapse.

## Successes / failures

Localization wins in 9–10/10 cats (dowel +.30, rope +.13, cookie +.21
AUPRO). Image-level negative transfer persists: cable_gland (.802→.711),
rope (.813→.697), foam (.790→.666), tire, bagel vs fixed. Defects:
potato/contamination .352→.666 recovered; rope/open .842→.875; carrot/hole
.862→.809 and cookie/contamination still weak.

## Remaining gaps (motivate E5/E6/E7)

Image-level transfer still negative in 4–5 cats; concat is undirected and
cannot control which modality informs which; RGB-absorbs-more asymmetry
suggests directed query-based exchange is the right next test.
