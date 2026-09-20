# Verified Statistical Significance & Hypothesis Testing Report

**Document**: `docs/PNTC_STATISTICAL_REPORT_VERIFIED.md`  
**Purpose**: Formal statistical hypothesis testing evaluating PNTC vs H4B across all 10 MVTec 3D-AD categories.

---

## 1. Experimental Design & Statistical Formulation

- **Experimental Unit**: Dataset Category ($N = 10$ distinct physical object classes).
- **Paired Observations**: For each category $c \in \{1, \dots, 10\}$, we evaluate paired metric values $(y_{\text{H4B}, c}, y_{\text{PNTC}, c})$.
- **Metric Difference**: $D_c = y_{\text{PNTC}, c} - y_{\text{H4B}, c}$.
- **Null Hypothesis ($H_0$)**: The true mean difference across categories is zero ($\mu_D = 0$).
- **Alternative Hypotheses**:
  - **Two-Sided ($H_1$)**: $\mu_D \neq 0$ (PNTC performance differs from H4B).
  - **One-Sided ($H_1^+$)**: $\mu_D > 0$ (PNTC strictly improves over H4B).

---

## 2. Category-Level Paired Hypothesis Tests ($N = 10$)

### 2.1 Image-Level AUROC (I-AUROC)
- **Mean Delta ($\bar{D}$)**: $+0.01140 \pm 0.01179$
- **Standard Error ($\text{SE}$)**: $0.00373$
- **95% Confidence Interval**: $[+0.00297, \, +0.01983]$
- **Effect Size (Cohen's $d$)**: $0.967$ (Large effect)
- **Parametric Paired t-Test ($df = 9$)**:
  - $t = 3.0585$
  - Two-Sided $p = \mathbf{0.013606}$ ($p < 0.05 \implies \text{Statistically Significant}$)
  - One-Sided $p = \mathbf{0.006803}$ ($p < 0.01 \implies \text{Statistically Significant}$)
- **Non-Parametric Wilcoxon Signed-Rank Test ($n_{\text{nonzero}} = 8$)**:
  - $W = 0.0$ (All 8 non-zero delta categories strictly improved)
  - Two-Sided $p = \mathbf{0.007812}$ ($p < 0.01$)
  - One-Sided $p = \mathbf{0.003906}$ ($p < 0.01$)

### 2.2 Pixel-Level AUROC (P-AUROC)
- **Mean Delta ($\bar{D}$)**: $+0.00307 \pm 0.00474$
- **Standard Error ($\text{SE}$)**: $0.00150$
- **95% Confidence Interval**: $[-0.00032, \, +0.00646]$
- **Effect Size (Cohen's $d$)**: $0.647$ (Medium-large effect)
- **Parametric Paired t-Test ($df = 9$)**:
  - $t = 2.0471$
  - Two-Sided $p = \mathbf{0.070950}$
  - One-Sided $p = \mathbf{0.035475}$ ($p < 0.05$)
- **Non-Parametric Wilcoxon Signed-Rank Test ($n_{\text{nonzero}} = 9$)**:
  - $W = 0.0$ (All 9 non-zero delta categories strictly improved)
  - Two-Sided $p = \mathbf{0.003906}$ ($p < 0.01$)
  - One-Sided $p = \mathbf{0.001953}$ ($p < 0.01$)

### 2.3 Pixel-Level Localization (AUPRO@0.3)
- **Mean Delta ($\bar{D}$)**: $+0.01087 \pm 0.01056$
- **Standard Error ($\text{SE}$)**: $0.00334$
- **95% Confidence Interval**: $[+0.00332, \, +0.01842]$
- **Effect Size (Cohen's $d$)**: $1.029$ (Very large effect)
- **Parametric Paired t-Test ($df = 9$)**:
  - $t = 3.2555$
  - Two-Sided $p = \mathbf{0.009909}$ ($p < 0.01 \implies \text{Statistically Significant}$)
  - One-Sided $p = \mathbf{0.004955}$ ($p < 0.01 \implies \text{Statistically Significant}$)
- **Non-Parametric Wilcoxon Signed-Rank Test ($n_{\text{nonzero}} = 10$)**:
  - $W = 0.0$ (All 10 categories strictly improved)
  - Two-Sided $p = \mathbf{0.001953}$ ($p < 0.01$)
  - One-Sided $p = \mathbf{0.000977}$ ($p < 0.001$)

---

## 3. Distinction: Coreset Initialization Variance vs Category Generalization

1. **Multi-Seed Variance ($N_{\text{seeds}} = 3$)**:
   - Evaluates prototype sampling stability within the same dataset.
   - Standard deviation across seeds is $< 0.00003$, proving that greedy coreset selection is deterministic and stable.
   - Note: $N=3$ random seeds is an index stability check and is not used as the basis for statistical significance claims.
2. **Category Generalization ($N_{\text{categories}} = 10$)**:
   - Evaluates generalization across diverse physical object geometries and textures.
   - The Wilcoxon signed-rank test confirms statistically significant improvements across Image AUROC ($p=0.0078$), Pixel AUROC ($p=0.0039$), and AUPRO ($p=0.0020$).
