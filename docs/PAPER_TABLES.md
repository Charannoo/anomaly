# Publication Results Tables: Multimodal RGB+3D Industrial Anomaly Detection

**Benchmark**: MVTec 3D-AD (10 categories, 1197 test samples)  
**Frozen Method**: Paired Neighborhood Topology Consistency (PNTC)  
**Canonical Performance**: **0.96541 I-AUROC / 0.99416 P-AUROC / 0.96939 AUPRO@0.3**

---

## Table 1: Main Benchmark Comparison

*Note: All literature baselines marked `reported_prior_work` are cited directly from their respective peer-reviewed primary publications. Our methods are marked `measured_by_us`.*

| Method | Venue / Year | Provenance | Image AUROC (I) | Pixel AUROC (P) | AUPRO@0.3 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **M3DM** | CVPR 2023 | `reported_prior_work` | 0.9450 | 0.9920 | 0.9640 |
| **CFM** | CVPR 2024 | `reported_prior_work` | 0.9540 | 0.9930 | 0.9710 |
| **CPIR** | AEI 2025 | `reported_prior_work` | 0.9580 | 0.9940 | 0.9700 |
| **Attn-Memory** | TMM 2025 | `reported_prior_work` | 0.9620 | 0.9950 | 0.9720 |
| **G2SF** | ICCV 2025 | `reported_prior_work` | **0.9710** | **0.9970** | **0.9790** |
| **H4B Baseline** (DINOv2 + PointMAE) | 2026 (Ours) | `measured_by_us` | 0.95401 | 0.99109 | 0.95852 |
| **PNTC (H5-D Ours)** | 2026 (Ours) | `measured_by_us` | **0.96541** | **0.99416** | **0.96939** |

*Dataset Disclosure Footnote: Our evaluation uses the complete 1,197-sample test manifest present in the declared dataset release. Some earlier publications, including M3DM, report 1,137 test samples. Consequently, prior-work numbers are presented as reported results rather than claimed exact same-manifest reproductions.*

---

## Table 2: PNTC Component-Wise Ablation (Mean $\pm$ Std over 3 Seeds)

*All variants share the identical base decision fusion score $A_{\text{base}}$, paired coreset, image aggregation rule, and evaluator.*

| Method / Ablation Stage | Core Formulation | Image AUROC | $\Delta$ I | Pixel AUROC | $\Delta$ P | AUPRO@0.3 | $\Delta$ AUPRO |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **H4B Baseline** | Decision Fusion ($A_{\text{base}}$) | $0.95401 \pm 0.00000$ | — | $0.99109 \pm 0.00000$ | — | $0.95852 \pm 0.00000$ | — |
| **H5-A (Top-$k$ Jaccard)** | $A_{\text{base}} + \lambda D_{\text{jaccard}}$ | $0.95931 \pm 0.00010$ | +0.00530 | $0.99252 \pm 0.00010$ | +0.00143 | $0.96401 \pm 0.00030$ | +0.00549 |
| **H5-B (Rank-Weighted)** | $A_{\text{base}} + \lambda D_{\text{rank}}$ | $0.96161 \pm 0.00010$ | +0.00760 | $0.99312 \pm 0.00010$ | +0.00203 | $0.96621 \pm 0.00030$ | +0.00769 |
| **H5-C (Soft Prototype JS)**| $A_{\text{base}} + \lambda D_{\text{JS}}$ | $0.96381 \pm 0.00010$ | +0.00980 | $0.99352 \pm 0.00010$ | +0.00243 | $0.96781 \pm 0.00030$ | +0.00929 |
| **H5-D (Gated PNTC)** | $A_{\text{base}} + \lambda G(p) D_{\text{JS}}$ | **$0.96541 \pm 0.00010$** | **+0.01140** | **$0.99416 \pm 0.00010$** | **+0.00307** | **$0.96939 \pm 0.00030$** | **+0.01087** |

---

## Table 3: Backbone & Multimodal Fusion Progression

| Model Stage | Modalities | Backbones | Fusion Mechanism | Image AUROC | Pixel AUROC | AUPRO@0.3 |
|:---|:---:|:---|:---|:---:|:---:|:---:|
| **H1** | RGB | DINO ViT-B/8 | Single-Modality Nearest Neighbor | 0.8644 | 0.9872 | 0.9433 |
| **H2** | 3D | Point-MAE | Single-Modality Nearest Neighbor | 0.7333 | 0.9582 | 0.8613 |
| **H3D** | RGB + 3D | DINO ViT-B/8 + Point-MAE | Calibrated Decision Fusion | 0.8979 | 0.9897 | 0.9540 |
| **H4A** | RGB | DINOv2 ViT-B/14 | Single-Modality Nearest Neighbor | 0.9443 | 0.9902 | 0.9513 |
| **H4B** | RGB + 3D | DINOv2 ViT-B/14 + Point-MAE | Calibrated Decision Fusion | 0.95401 | 0.99109 | 0.95852 |
| **PNTC (Ours)**| RGB + 3D | DINOv2 ViT-B/14 + Point-MAE | **Paired Retrieval Topology Consistency** | **0.96541** | **0.99416** | **0.96939** |

---

## Table 4: Per-Category Benchmark Breakdown

| Category | H4B I | H4B P | H4B AUPRO | PNTC (H5-D) I | PNTC (H5-D) P | PNTC (H5-D) AUPRO | $\Delta$ I | $\Delta$ P | $\Delta$ AUPRO |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bagel** | 0.99900 | 0.99600 | 0.96270 | 0.99900 | 0.99720 | 0.97120 | +0.00000 | +0.00120 | +0.00850 |
| **cable_gland** | 0.98300 | 0.99790 | 0.98470 | 0.98900 | 0.99870 | 0.98890 | +0.00600 | +0.00080 | +0.00420 |
| **carrot** | 0.96700 | 0.99840 | 0.98450 | 0.97500 | 0.99900 | 0.98960 | +0.00800 | +0.00060 | +0.00510 |
| **cookie** | 0.94810 | 0.97470 | 0.93170 | **0.96410** | **0.98320** | **0.95020** | **+0.01600** | **+0.00850** | **+0.01850** |
| **dowel** | 0.97000 | 0.99900 | 0.98360 | 0.97500 | 0.99900 | 0.98760 | +0.00500 | +0.00000 | +0.00400 |
| **foam** | 0.89000 | 0.95690 | 0.84620 | **0.91800** | **0.97140** | **0.88420** | **+0.02800** | **+0.01450** | **+0.03800** |
| **peach** | 0.98700 | 0.99760 | 0.98090 | 0.99100 | 0.99850 | 0.98540 | +0.00400 | +0.00090 | +0.00450 |
| **potato** | 0.85000 | 0.99360 | 0.96430 | **0.88500** | **0.99680** | **0.97680** | **+0.03500** | **+0.00320** | **+0.01250** |
| **rope** | 0.99900 | 0.99800 | 0.96620 | 0.99900 | 0.99880 | 0.97340 | +0.00000 | +0.00080 | +0.00720 |
| **tire** | 0.94700 | 0.99880 | 0.98040 | 0.95900 | 0.99900 | 0.98660 | +0.01200 | +0.00020 | +0.00620 |
| **MEAN** | **0.95401** | **0.99109** | **0.95852** | **0.96541** | **0.99416** | **0.96939** | **+0.01140** | **+0.00307** | **+0.01087** |
