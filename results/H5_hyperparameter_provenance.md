# Phase H5 Hyperparameter Provenance & Leakage Audit

**Audit Date**: September 2026  
**Artifact**: `results/H5_hyperparameter_provenance.md`  
**Purpose**: Full provenance audit of all hyperparameters and constants used in H5 PNTC to verify zero test-label leakage and zero post-hoc tuning on MVTec test sets.

---

## 1. Hyperparameter Inventory Table

| Hyperparameter | Symbol | Value | Where Defined | How Selected | Data Split Used | Test Label Leakage? |
|---|---|---|---|---|---|:---:|
| Topology Fusion Weight | $\lambda$ | $0.35$ | `docs/H5_PNTC_FORMULATION.md` | Predeclared scaling ratio relative to standardized normal score variance | Train-Normal Calibration | **NO** |
| RGB Soft Prototype Temperature | $\tau_{\text{rgb}}$ | Derived (~0.42) | `run_h5_pntc_eval.py` | Median 1-NN distance across normal training features | Train-Normal Only | **NO** |
| 3D Soft Prototype Temperature | $\tau_{\text{xyz}}$ | Derived (~0.51) | `run_h5_pntc_eval.py` | Median 1-NN distance across normal training features | Train-Normal Only | **NO** |
| Neighborhood Size | $k$ | $5$ | `run_h5_pntc_eval.py` | Predeclared standard nearest-neighbor retrieval depth | Predeclared Constant | **NO** |
| Coreset Budget | $M$ | $15{,}000$ | `cache_utils.py` | Hardware memory cap established in H1/H2 | Train-Normal Only | **NO** |
| Coreset Random Seeds | $s$ | $[42, 100, 2026]$ | `run_h5_pntc_eval.py` | Standard multi-seed verification set | Deterministic Seed List | **NO** |
| Distribution Smoothing Floor | $\epsilon$ | $10^{-6}$ | `run_h5_pntc_eval.py` | Numerical stability epsilon for categorical JS divergence | Fixed Constant | **NO** |
| Normal Standardization Means | $\mu_{\text{train, rgb}}, \mu_{\text{train, xyz}}$ | Empirical per cat | `run_h5_pntc_eval.py` | Sample mean of 1-NN distances on nominal training set | Train-Normal Only | **NO** |
| Normal Standardization Stds | $\sigma_{\text{train, rgb}}, \sigma_{\text{train, xyz}}$ | Empirical per cat | `run_h5_pntc_eval.py` | Sample std of 1-NN distances on nominal training set | Train-Normal Only | **NO** |
| Dispersion Scale | $\bar{\sigma}_{\text{dispersion}}$ | Empirical per cat | `run_h5_pntc_eval.py` | Average top-$k$ distance variance across nominal training set | Train-Normal Only | **NO** |
| Image Score Aggregator | $\text{Agg}$ | $\text{mean\_top\_0.5\%}$ | Frozen from H3.5 | Selected in H3.5 robust aggregation evaluation | Frozen Baseline Rule | **NO** |
| Pixel Smoothing Kernel | $\sigma_{\text{blur}}$ | $4$ | Frozen from H1 | Standard PatchCore/M3DM Gaussian spatial filter | Fixed Protocol | **NO** |

---

## 2. Leakage Audit Check

1. **Category-Specific Anomaly Tuning**: **NONE**. All hyperparameters ($\lambda$, $k$, $M$, $\text{Agg}$) are global constants applied uniformly across all 10 categories.
2. **Post-Hoc Metric Optimization**: **NONE**. No parameter was adjusted after observing test set AUROC or AUPRO.
3. **Data Splitting**: Coreset selection, temperature calibration ($\tau$), and normalization parameters ($\mu, \sigma$) are computed exclusively on `train/good` samples.
4. **Conclusion**: The Phase H5 evaluation strictly satisfies zero-leakage research standards.
