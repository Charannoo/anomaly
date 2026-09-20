# PHASE4B — Attention Ablation Decision (frozen)

Status: **FROZEN — model selection closed.** Phase 4B is complete; the
cross-modal attention mechanism (E5/E6/E7) was tested honestly and does **not**
become the XMV-AD core. E4 (residual feature concatenation) is the final model:
**"XMV-AD E4"** / **"XMV-AD Residual Fusion"**. E7 is recorded and cited as
**"XMV-Attention Ablation / E7"**.

This is a decision document, not a re-run protocol. All numbers below come from
already-frozen tables (`experiments/tables/*`) produced under the identical
Phase 4 protocol; nothing was re-tuned after test evaluation.

## 1. What was compared

| Model | Interaction | Trainable params | Checkpoint |
|---|---|---|---|
| E4 concat | 1/16 feature concatenation + zero-init residual gates | 8738 | `checkpoints/E4_concat_fusion/` |
| E5 | RGB <- Depth (RGB queries depth) | 12770 | `checkpoints/E5_cross_attn_rgb/` |
| E6 | Depth <- RGB (depth queries RGB) | 12770 | `checkpoints/E6_cross_attn_depth/` |
| E7 | bidirectional (RGB <-> Depth) | 12770 | `checkpoints/E7_xmv_bidirectional/` |

Fairness was held fixed: same frozen Phase 2 RGB/depth branches, same teachers,
same normal-only train/val splits, 256px input, fusion at 1/16 only, AdamW
1e-3/wd 1e-4, batch 8, 50 epochs, early-stop patience 8, seed 42, same metric
code, E3-declared calibration **refit per model** on its own validation-normal
maps, calibrated 50/50 mean fusion, `max` primary readout.

## 2. Identity control

gamma=0 reproduces Phase 2 bitwise (`torch.equal`, all three modes), and
attention-diag entropy/mass stats were captured per experiment
(`E5E6E7_attention_diag.csv`). Only the interaction module differs between E4,
E5, E6, E7.

## 3. Headline result (MEAN over 10 categories, calibrated 50/50, max)

| Method | I-AUROC | I-AUROC(top1%) | P-AUROC | P-AUPRC | AUPRO |
|---|---|---|---|---|---|
| E4 concat | **.6536** | **.7516** | **.9562** | **.2075** | **.8530** |
| E5 RGB<-Depth | .6277 | .7222 | .9168 | .1029 | .7278 |
| E6 Depth<-RGB | .6276 | .7367 | .9152 | .1045 | .7237 |
| E7 bidirectional | .6337 | .7352 | .9179 | .1056 | .7312 |

E7 minus E4: I-AUROC **-0.0199**, P-AUROC **-0.0383**, AUPRO **-0.1218**.
E7 vs fixed fusion (E3A): I-AUROC -0.0062, P-AUROC +0.0031, AUPRO +0.0088.

**Conclusion (frozen):** adding cross-modal attention at 1/16 recovers only
"fixed-fusion-level" behavior and loses most of E4's localization gain. The
decisive signal is AUPRO: E7 .7312 vs E4 .8530 (Δ -0.1218). Attention is a
published ablation, not the product.

## 4. Bootstrap (image-level, per-category + pooled)

`E7_vs_E4_bootstrap.csv`, 1000 resamples/category, Δ = E7 minus E4:

- POOLED Δ = **-0.0399**, 95% CI [**-0.0687, -0.0124**], excludes zero.
- Per-category: significant only foam +0.1019 (CI [0.005, 0.1963]) and
  peach **-0.2398** (CI [-0.3866, -0.0972]); the other 8 categories are not
  distinguishable from zero.

The pooled test says E7 is *worse* than E4 at image level; the two significant
categories point in opposite directions. No credible deficiency of E4.

## 5. Directionality

`E5E6E7_directionality.csv` (best of E5/E6 one-way vs E7 bidirectional; metric
= I-AUROC/P-AUROC/AUPRO):

- Bidirectional equals or exceeds the best one-way direction in every
  localization metric (bidir-minus-best-oneway ≤ 0 on P-AUROC/AUPRO for all
  categories), i.e. E7 beats E5/E6 on localization — but never reaches E4.
- Best *direction* (image-level) differs per category (E5 on 4, E6 on 4, E7 on
  2); there is no consistent dominant direction, which is explainable: RGB and
  depth topographies are spatially registered so information flows in both
  directions are complements, not either/or.

## 6. Trained gate magnitudes

`E5E6E7_gamma.csv`. E5/E6/E7 gates settle at |gamma| in ~[0.10, 0.35] with
mixed signs (e.g. E5 cable_gland -0.17, E6 dowel -0.22), i.e. the learned
residual is a real but small perturbation, sometimes *suppressive*. Residual
magnitudes in `E5E6E7_attention_diag.csv` (res~0.15-0.35) and attention
entropies ~3.8-5.4 nats (no collapse). Nothing here rescues the headline.

## 7. Routing and validity behavior

All three attention modes preserve the Phase 4 validity routing: RGB<-Depth
masks invalid depth keys (additive -1e9, all-zero-weight guard); Depth<-RGB
masks invalid-depth queries; zero-valid samples receive zero residual.
`valid_depth_attn_mass = 1.0` everywhere; invalid depth never leaks into the
attention path.

## 8. Fixed decision

1. **E4 is the final fusion core** — "XMV-AD E4" / "XMV-AD Residual Fusion".
   All Phase 5 explainability, paper tables and figures are E4-based.
   Further model iteration is out of scope (INT8/ONNX/OpenVINO/dashboard/new
   architectures are explicitly deferred / excluded).
2. **E7 is "XMV-Attention Ablation / E7"** — reported as a published ablation
   with its honest negative result, per Section 3-4.
3. Attention is not removed from the record; it is demoted. Any future claim
   must reproduce the pooled negative result, not hide it.
4. No test-set re-tuning occurred. All Phase 5 work operates on frozen E4
   checkpoints and already-computed test predictions
   (`experiments/predictions/E4_concat_fusion/`).

## 9. Inputs to Phase 5

- E4 checkpoints: `checkpoints/E4_concat_fusion/<cat>_best.pt`
- E4 calibration (validation-normal robust median/MAD, `E4_concat_calibration.json`)
- E4 test predictions (per-image `rgb_map/depth_map/map_mean/map_max/gt/label/defect`).
- E4 model/params/latency: 8738 params, ~42 KB checkpoint, 34.0 ms mean CPU (batch 1, 256px) — see `E5E6E7_dev_latency.csv`.