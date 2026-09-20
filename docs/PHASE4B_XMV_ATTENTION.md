# PHASE4B — E5/E6/E7 directional cross-attention + bidirectional XMV fusion

Phase 4B introduces the core XMV cross-modal interaction mechanism as a single
generic module configured by direction, and compares it head-to-head against the
frozen E4 concatenation baseline under an identical protocol.

- **E5** `rgb_from_depth`: RGB queries Depth only.
- **E6** `depth_from_rgb`: Depth queries RGB only.
- **E7** `bidirectional`: both directions (the proposed XMV interaction block).

Code: `src/xmvad/models/fusion/cross_attention.py` (`XMVAttention`, `XMVModel`),
`scripts/train_cross_attention.py`, `scripts/evaluate_cross_attention.py`,
configs `configs/cross_attention_rgb_from_depth.yaml`,
`configs/cross_attention_depth_from_rgb.yaml`, `configs/xmv_ad.yaml`.
Tests: `tests/test_attention.py` (20 tests).

## Architecture (exact)

One module family; configuration selects direction. At 1/16 only (N=256 tokens
at 256px, direct attention, no windowing):

- Per modality: `Conv1x1(48->32)` -> LayerNorm **inside the attention path**
  only (residual branch untouched, so gamma=0 is exact identity).
- `attention_dim=32`, `num_heads=2`, `head_dim=16` (pre-registered; no sweep).
- `Q/K/V` are `Linear(32,32)`; attention is scaled dot-product; heads merged;
  `Wo: 32->48` -> reshape `[B,48,16,16]`.
- Residual `S_x = S + gamma * delta`; `gamma` trainable scalar, init 0.
- **No** self-attention, FFN, transformer stack, or positional parameters
  (RGB/depth are spatially registered; aligned token ordering is the position
  signal — documented choice).
- Modality identity preserved: `S_rgb_x` / `S_depth_x` stay separate and are
  scored against their own frozen teachers (no pre-scoring collapse).
- Validity: RGB<-Depth masks invalid depth **keys** pre-softmax (additive
  `-1e9`, all-zero-weight guard, NaN-proof); samples with zero valid keys get a
  zero residual. Depth<-RGB masks `delta_depth` at invalid depth **queries**.
- Trainable params: **12770** (E7; E5/E6 use the same module, one direction
  inactive — inactive params receive no gradient).

## Fairness (identical to E4)

Same frozen Phase 2 RGB/depth branches, teacher targets, normal-only train +
validation splits, input 256, 1/16 only, AdamW 1e-3 / wd 1e-4, batch 8, 50
epochs, early stopping patience 8 (val loss w/o residual penalty), seed 42,
identical metric code, E3-declared calibration refit per experiment on its own
validation-normal maps, final calibrated 50/50 mean (invalid -> RGB only),
`max` primary readout. Only the interaction module differs.

## Identity control

gamma=0 reproduces Phase 2 exactly (bitwise `torch.equal` on RGB + depth maps,
all three modes), plus feature-level identity. Verified in tests before training.

## Results (test, calibrated 50/50 mean, max primary)

MEAN over 10 categories:

| Method | I-AUROC(max) | I-AUROC(top1%) | I-AUPRC(max) | P-AUROC | P-AUPRC | AUPRO |
|---|---|---|---|---|---|---|
| RGB (E1) | .6401 | .6862 | .8630 | .8950 | .0698 | .6855 |
| Depth (E2) | .5340 | .5875 | .8166 | .7542 | .0489 | .4918 |
| Fixed fusion (E3A) | .6400 | .7321 | — | .9148 | — | .7224 |
| E4 concat | **.6536** | **.7516** | **.8699** | **.9562** | **.2075** | **.8530** |
| E5 RGB<-Depth | .6277 | .7222 | .8647 | .9168 | .1029 | .7278 |
| E6 Depth<-RGB | .6276 | .7367 | .8667 | .9152 | .1045 | .7237 |
| E7 bidirectional | .6337 | .7352 | .8680 | .9179 | .1056 | .7312 |

E7 - E4: I-AUROC **-0.0199**, I-AUPRC -0.0019, P-AUROC **-0.0383**,
P-AUPRC **-0.1019**, AUPRO **-0.1218**.
E7 - Fixed: I-AUROC -0.0062, top1% +0.0031, P-AUROC +0.0031, AUPRO +0.0088.
E7 - BestSingle: +0.0064 I, +0.0490 top1%, +0.0229 P-AUROC, +0.0457 AUPRO.

Full per-category table: `experiments/tables/E7_vs_all.csv`; per-experiment
`E5_cross_attn_rgb_metrics.csv`, `E6_cross_attn_depth_metrics.csv`,
`E7_xmv_bidirectional_metrics.csv`.

**E7 does not beat E4.** It is comparable to (slightly above) fixed fusion and
well below E4 on localization. E5/E6 are similar to E7, slightly lower.

## Directionality (§20)

Best direction per category is split, no single direction dominates
(`E5E6E7_directionality.csv`):

- E5 (RGB<-Depth) best I-AUROC on bagel (.885), cable_gland (.747), foam (.775),
  tire (.579).
- E6 (Depth<-RGB) best I-AUROC on cookie (.533), dowel (.639), peach (.415).
- E7 best on carrot (.734), potato (.550), rope (.698) and on most pixel metrics.
- Bidirectional minus best one-way is small (|d| <= .027 on I-AUROC, <= .010
  localization) — directionality is weak.

Wording: "RGB<-Depth interaction improved metric X"; no physical-causality
claims.

## Map change (§22)

E7 changes modality maps far less than E4 (`E7_xmv_bidirectional_map_change.csv`
vs `E4_map_change.csv`):

- E7 vs Phase 2: RGB mean|d| 0.005-0.018, corr 0.976-0.995; depth mean|d|
  0.003-0.006, corr 0.988-0.999.
- E4 vs Phase 2: RGB mean|d| 0.021-0.075, corr 0.687-0.923; depth mean|d|
  0.010-0.029, corr 0.834-0.976.
- E7 vs E4: RGB corr 0.692-0.927, depth corr 0.844-0.975.

E7 learned much smaller residuals (gammas .03-.23 vs E4 .22-.38) with diffuse
attention, so it stays close to the frozen representation and to fixed fusion.

## Cross-modal influence / attention diagnostics (§23)

`E5E6E7_attention_diag.csv`, `E5E6E7_gamma.csv`:

- Gammas non-zero and stable (no explosion): E5 gamma_rd .027-.11 (negative
  cable/foam/tire); E6 gamma_dr .12-.23 (negative dowel/foam); E7 both
  directions .10-.21. Much smaller than E4.
- Residual magnitudes: RGB .15-.35, depth .10-.24.
- Attention entropy 3.8-4.9 nats (uniform = 5.545) — diffuse but not fully
  collapsed; some E5/E6 categories approach uniform (~4.9).
- Valid-depth attention mass = 1.0 (masking renormalizes over valid keys); no
  NaNs, no single-token collapse.

## Negative transfer (§24)

- Recovered vs E4: foam image-level (E7 .768 vs E4 .666; E5 .775), cable_gland
  image-level (E7 .720 vs E4 .711), bagel/rope/potato I-AUROC modestly up vs E4.
- Worsened vs E4: peach I-AUROC .330 vs .570 (bootstrap significant), carrot
  AUPRO .827 vs .948, dowel AUPRO .625 vs .873, cookie I-AUROC .466 vs .571.
- Defect cases (`E5E6E7_defect_level.csv`): bagel/crack .953 (preserved);
  foam/color .828 (preserved); rope/open .844; potato/contamination .411 vs E4
  .666 (E4's gain not held); carrot/hole .858 (preserved); cookie/contamination
  .314 (still weak). E7 improves foam/combined (.795 vs .748) and
  potato/hole (.727 vs .490) vs E4.

## Statistical comparison (§25)

Paired image-level bootstrap (1000 resamples, `E7_vs_E4_bootstrap.csv`):

- POOLED delta E7-E4 = **-0.0399**, 95% CI [-0.0687, -0.0124] — significantly
  worse than E4 at image level.
- foam +0.1019 (CI [.005, .196], E7 better); peach -0.2398 (CI [-.387, -.097],
  E7 worse); all other categories CI includes zero.

## Computational analysis (§26, dev only)

`E5E6E7_dev_latency.csv` (batch 1, 256px, same machine, CPU):

| Model | params | ckpt KB | predict mean ms |
|---|---|---|---|
| E4 concat | 8738 | 42.3 | 38.3 |
| E5 | 12770 | 60.2 | 40.2 |
| E6 | 12770 | 60.2 | 39.8 |
| E7 | 12770 | 60.2 | 42.4 |

All < 20k params; overhead small (~4 ms over E4).

## Training diagnostics (§27)

Per-epoch train/val loss, gammas, residual magnitude, attention entropy logged
to `experiments/runs/<stamp>_<exp>_<cat>/training_log.csv` + `training_curves.png`
(30 runs). No gamma explosion, no NaN attention, no uniform/single-token
collapse.

## Tests (§28)

20 attention tests (shapes, head split/merge, all three modes, invalid-key
masking, invalid-query masking, all-invalid fallback, NaN safety, gamma=0
identity, frozen branches, gradient routing active/inactive, deterministic init,
save/load, map sizes, param budget, no-test-in-training source guard). Full
suite: **84 passed, 1 skipped** (ONNX/Phase 7 placeholder).

## Recommendation (§29/§30)

**Option 2 governs the primary decision, with Option 3 as a bounded
correction path.** E7 does not beat E4; the evidence shows lightweight
concatenation is sufficient at this scale and cross-attention at 1/16 with 32
dims/diffuse attention does not justify its complexity. E7 ≈ fixed fusion on
localization, below E4. It does improve specific hard categories (foam,
cable_gland image-level). If attention is pursued, one controlled correction is
justified (sharper attention / stronger residual / higher dim) — not part of
this phase.

State: **PHASE4B_COMPLETE — awaiting review.** No INT8/ONNX/OpenVINO/dashboard,
no reliability-gate integration, no multi-scale attention, no unfreezing.
