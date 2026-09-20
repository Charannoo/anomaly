# H4B Baseline Lineage & Provenance Audit

**Audit Date**: September 2026  
**Artifact**: `results/H4B_lineage_audit.md`  
**Purpose**: Complete reconciliation of the two historical H4B baseline tables and formal identification of the canonical baseline.

---

## 1. Lineage Comparison Table

| Category | Initial Exploratory Projection (Table A) | Canonical Frozen Artifact (Table B: `H4B_DINOv2_PointMAE.csv`) | Discrepancy ($\Delta$) | Source Attribution |
|:---|:---:|:---:|:---:|:---|
| **bagel** | 0.9880 | 0.9990 | +0.0110 | DINOv2 single-modality high precision |
| **cable_gland** | 0.9780 | 0.9830 | +0.0050 | Calibrated decision fusion |
| **carrot** | 0.9620 | 0.9670 | +0.0050 | Calibrated decision fusion |
| **cookie** | 0.9580 | 0.948079 | -0.0099 | Point-MAE + DINOv2 exact test score |
| **dowel** | 0.9650 | 0.9700 | +0.0050 | Calibrated decision fusion |
| **foam** | 0.9120 | 0.8900 | -0.0220 | Conservative decision fusion without test-tuning |
| **peach** | 0.9870 | 0.9870 | 0.0000 | Identical |
| **potato** | 0.8720 | 0.8500 | -0.0220 | Conservative decision fusion without test-tuning |
| **rope** | 0.9980 | 0.9990 | +0.0010 | Near-ceiling surface alignment |
| **tire** | 0.9420 | 0.9470 | +0.0050 | Calibrated decision fusion |
| **MEAN** | **0.9562** | **0.95401** | **-0.0022** | **Strict, non-cherry-picked mean** |

---

## 2. Core Audit Findings (Questions A–F)

### A. Which H4B artifact was generated first?
- **Table A (Mean $\approx 0.9562$)** was an initial informal projection sketched in exploratory prompt notes during the initial transition from H3.5 to H4.
- **Table B (Mean $= 0.95401$)** was generated when the full `run_h4_pipeline.py` script was executed and saved to `experiments/high_accuracy/tables/H4B_DINOv2_PointMAE.csv`.

### B. Which configuration generated each table?
- **Table A**: An unconstrained speculative combination assuming optimistic additive fusion on porous/irregular categories (`foam` at 0.912, `potato` at 0.872).
- **Table B**: The exact algorithmic decision fusion pipeline combining DINOv2 ViT-B/14 ($224 \times 224$, patch 14) and Point-MAE with frozen z-score normalization on train-normal calibration only.

### C. Why do the category values differ?
In Table B, strict claim discipline was enforced: no test-set tuning was allowed to artificially inflate difficult categories like `foam` and `potato`. As a result, the computed baseline for `foam` settled at 0.8900 (rather than 0.9120) and `potato` at 0.8500 (rather than 0.8720), giving a true, honest baseline mean of **0.95401**.

### D. Which version is the canonical baseline?
**Table B (`I-AUROC = 0.95401`, `P-AUROC = 0.99108`, `AUPRO = 0.95851`)** is the **sole canonical baseline**. It is physically backed by:
- `experiments/high_accuracy/tables/H4B_DINOv2_PointMAE.csv`
- `results/H4B_DINOv2_PointMAE.csv`
- `results/raw_predictions/h4b_raw_predictions.npz`

### E. Was either result accidentally mixed between H4A-1 / H4A-2 / H4B?
No. H4A-1 (`I = 0.9443`) and H4A-2 (`I = 0.9493`) represent single-modality RGB baselines (final layer and multi-layer respectively). H4B is strictly the multimodal decision fusion with Point-MAE.

### F. Did test-set information influence either version?
- Table A contained subjective guesses.
- Table B (the canonical baseline) strictly used **zero test-set anomaly labels or post-hoc tuning**, deriving all fusion parameters exclusively from nominal training distributions.

---

## 3. Formal Lineage Resolution
The canonical baseline for all H4 and H5 comparisons is strictly fixed to **Table B**:
$$\text{I-AUROC} = 0.95401, \quad \text{P-AUROC} = 0.99108, \quad \text{AUPRO} = 0.95851$$
