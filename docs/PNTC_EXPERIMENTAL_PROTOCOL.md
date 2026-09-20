# Canonical Experimental Protocol & Benchmark Specification

**Document**: `docs/PNTC_EXPERIMENTAL_PROTOCOL.md`  
**Dataset**: MVTec 3D-AD (10 Industrial Categories, 1197 Test Samples)

---

## 1. Dataset & Split Specifications

- **Dataset**: MVTec 3D-AD (Industrial 3D Anomaly Detection Benchmark).
- **Categories (10)**: `bagel`, `cable_gland`, `carrot`, `cookie`, `dowel`, `foam`, `peach`, `potato`, `rope`, `tire`.
- **Sample Distribution**:
  - **Training (Nominal only)**: 2,654 nominal items across all categories.
  - **Validation**: Held-out nominal validation sets for parameter calibration.
  - **Test**: Exactly **1,197 test items** (comprising both nominal `good` items and authentic defect instances across all defect subcategories).

### Dataset-Protocol Disclosure
> **Dataset Disclosure**: *"Our evaluation uses the complete 1,197-sample test manifest present in the declared dataset release. Some earlier publications, including M3DM, report 1,137 test samples. Consequently, prior-work numbers are presented as reported results rather than claimed exact same-manifest reproductions."*

---

## 2. Preprocessing & Feature Extraction Pipelines

### 2.1 2D RGB Visual Modality
- **Backbone**: Frozen DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`).
- **Input Preprocessing**: Resized to $224 \times 224$, standard ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Patch Extraction**: $16 \times 16$ patch tokens ($N_{\text{tokens}}=256$, $D_{\text{rgb}}=768$).
- **Spatial Alignment**: Bilinearly interpolated to canonical spatial grid of $56 \times 56 = 3,136$ patches.

### 2.2 3D Point Cloud Modality
- **Backbone**: Frozen Point-MAE Transformer Encoder (`pointmae_pretrain.pth`, SHA256 `27ded932ff769d469f10cb99477e77b67b61a91cfca44ec2b25458021c385b0d`).
- **Input Preprocessing**: $N=2048$ surface points sampled via Farthest Point Sampling (FPS); coordinates zero-centered and normalized to unit sphere.
- **Feature Representation**: 12 Transformer encoder blocks, output token dimension $D_{\text{xyz}}=1152$.
- **Spatial Alignment**: Aligned to identical $56 \times 56$ spatial grid via nearest physical tile projection.

---

## 3. Memory Bank & Hyperparameter Configurations

| Parameter | Value | Scope / Selection Policy |
|:---|:---:|:---|
| **Coreset Budget ($M$)** | $15{,}000$ | Memory cap per category, MinMax subsampling on joint selector $q_j$ |
| **Coreset Random Seeds** | $[42, 100, 2026]$ | Deterministic multi-seed evaluation suite |
| **Nearest Neighbors ($k$)** | $5$ | Predeclared standard retrieval depth |
| **Topology Weight ($\lambda$)** | $0.35$ | Global constant derived on train-normal variance |
| **Temperature Rule ($\tau$)** | Empirical | Median 1-NN Euclidean distance on train-normal set |
| **Distribution Floor ($\epsilon$)** | $10^{-6}$ | Numerical stability constant for JSD |
| **Spatial Smoothing ($\sigma$)** | $3.8$ | Gaussian spatial filter applied before upsampling to $224 \times 224$ |
| **Image Aggregator** | `mean_top_0.5%` | Frozen rule established in H3.5 robust aggregation audit |

---

## 4. Metric Definitions & Software Stack

- **Image-Level AUROC**: Computed over all test sample anomaly scores via `sklearn.metrics.roc_auc_score(y_true, y_score)`.
- **Pixel-Level AUROC**: Computed over all valid flattened pixels across all test images via `sklearn.metrics.roc_auc_score(gt_flat, pred_flat)`.
- **AUPRO@0.3**: Official MVTec 3D-AD protocol with 8-connectivity connected components and trapezoid numerical integration up to $\text{FPR}=0.3$ normalized by $0.3$ (`src/xmvad/metrics/pro.py`).
- **Software Stack**: Python 3.13.6, PyTorch 2.14.0+cpu, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.1.
