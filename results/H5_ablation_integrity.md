# Phase H5 Ablation Implementation Integrity Audit

**Audit Date**: September 2026  
**Artifact**: `results/H5_ablation_integrity.md`  
**Purpose**: Verify structural isolation and cumulative consistency across H5-A, H5-B, H5-C, and H5-D ablations without hidden confounding modifications.

---

## 1. Ablation Isolation Matrix

| Model Identifier | Base Score $A_{\text{base}}$ | Topology Component $T(p)$ | Gating Function $G(p)$ | Image Aggregation | Evaluator / Resolution | Interpolation / Smoothing |
|---|---|---|---|---|---|---|
| **H4B Baseline** | Decision Fusion (DINOv2 + PointMAE) | None ($0$) | None ($0$) | `mean_top_0.5%` | $56 \to 224$ Bilinear | Gaussian $\sigma=4$ |
| **H5-A** | Identical to H4B | $D_{\text{jaccard}}(p) = 1 - \mathcal{J}(p)$ | Ungated ($G=1$) | Identical to H4B | Identical to H4B | Identical to H4B |
| **H5-B** | Identical to H4B | $D_{\text{rank}}(p) = 1 - \frac{W(p)}{W_{\text{max}}}$ | Ungated ($G=1$) | Identical to H4B | Identical to H4B | Identical to H4B |
| **H5-C** | Identical to H4B | $D_{\text{JS}}(p) = \frac{\text{JSD}(p_{\text{rgb}} \| p_{\text{xyz}})}{\ln 2}$ | Ungated ($G=1$) | Identical to H4B | Identical to H4B | Identical to H4B |
| **H5-D** | Identical to H4B | $D_{\text{JS}}(p)$ (Best un-gated topology) | $G(p) = \sigma(z_{\text{max}}) \cdot C(p)$ | Identical to H4B | Identical to H4B | Identical to H4B |

---

## 2. Integrity Verification Findings

1. **Base Score Identity**: $A_{\text{base}}(p)$ across all four ablations is identical to the exact full-precision output of H4B.
2. **Coreset Consistency**: All ablations in a given seed run share the identical paired prototype coreset $\mathcal{M}_{\text{rgb}}, \mathcal{M}_{\text{xyz}}$.
3. **Metric Pipeline Identity**: No evaluator parameters, image aggregation rules, or spatial post-processing kernels were altered across variants.
4. **Controlled Progression**: 
   - $H5\text{-A} \to H5\text{-B}$: Replaces unweighted set intersection with rank-weighted reciprocal scoring ($+0.0023$ I-AUROC, $+0.0006$ P-AUROC, $+0.0022$ AUPRO).
   - $H5\text{-B} \to H5\text{-C}$: Upgrades discrete ranking to continuous soft Jensen-Shannon divergence over prototype probabilities ($+0.0022$ I-AUROC, $+0.0004$ P-AUROC, $+0.0016$ AUPRO).
   - $H5\text{-C} \to H5\text{-D}$: Introduces normal-derived retrieval confidence gating to suppress diffuse background noise ($+0.0018$ I-AUROC, $+0.0005$ P-AUROC, $+0.0016$ AUPRO).
