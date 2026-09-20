# Prior-Work Literature Metric Audit

**Audit Date**: September 2026  
**Purpose**: Primary source verification of all external benchmark rows against original peer-reviewed conference and journal publications.

---

## 1. Primary Source Verification Table

| Method | Primary Venue | Year | Reported I-AUROC | Reported P-AUROC | Reported AUPRO@30 | Exact Source Location in Primary Paper | Dataset / Protocol Notes | Verified (Primary Paper) |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---:|
| **M3DM** (Wang et al.) | CVPR | 2023 | **0.9450** (94.5%) | **0.9920** (99.2%) | **0.9640** (96.4%) | Table 1 ("Quantitative results on MVTec 3D-AD") | Explicitly documents 1,137 test samples | **YES** |
| **CFM** (Costanzino et al.) | CVPR | 2024 | **0.9540** (95.4%) | **0.9930** (99.3%) | **0.9710** (97.1%) | Table 1 ("Anomaly Detection and Localization on MVTec 3D-AD") | Evaluates full MVTec 3D-AD benchmark | **YES (Corrected)** |
| **CPIR** (Zhang et al.) | AEI | 2025 | **0.9580** (95.8%) | **0.9940** (99.4%) | **0.9700** (97.0%) | Table 2 ("Comparison with SOTA methods") | Evaluates standard MVTec 3D-AD split | **YES** |
| **Attn-Memory** (Liu et al.) | IEEE TMM | 2025 | **0.9620** (96.2%) | **0.9950** (99.5%) | **0.9720** (97.2%) | Table 1 ("Performance comparison on MVTec 3D-AD") | Dual memory bank evaluation | **YES** |
| **G2SF** (Tao et al.) | ICCV | 2025 | **0.9710** (97.1%) | **0.9970** (99.7%) | **0.9790** (97.9%) | Table 1 ("Quantitative results on MVTec 3D-AD benchmark") | Anisotropic LSPN local metric | **YES** |

---

## 2. Fact-Check Corrections & Audit Notes

### 2.1 CFM (CVPR 2024) Correction
- **Previous Informal Entry**: `0.952 / 0.993 / 0.968`
- **Correct Primary Paper Entry**: `0.954 / 0.993 / 0.971` (Table 1 of Costanzino et al., CVPR 2024).
- **Comparison with PNTC (`0.96541 / 0.99416 / 0.96939`)**:
  - **I-AUROC**: $\text{PNTC } (0.96541) > \text{CFM } (0.95400)$ ($\mathbf{+0.01141}$)
  - **P-AUROC**: $\text{PNTC } (0.99416) > \text{CFM } (0.99300)$ ($\mathbf{+0.00116}$)
  - **AUPRO@30**: $\text{PNTC } (0.96939) < \text{CFM } (0.97100)$ ($\mathbf{-0.00161}$)

### 2.2 Dataset Manifest & Test Sample Count Note
- In M3DM (CVPR 2023), the paper text notes 1,137 test samples.
- Later works (including our local evaluation) evaluate on the complete 1,197-sample test set (10 categories, all defect subfolders + good).
- Consequently, all literature figures are documented strictly under `reported_prior_work` rather than identical-manifest local reproductions.
