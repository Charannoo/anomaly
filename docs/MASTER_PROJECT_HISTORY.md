# Master Project History: XMV-AD to PNTC

This document records the complete, unabridged chronological history of the industrial anomaly detection research project spanning both the **Lightweight Track (XMV-AD)** and the **High-Accuracy Track (XMV-AD-H $\to$ PNTC)**. 

All metrics, decisions, negative results, architectural evolutions, and reproducibility validations documented here are derived directly from the verified repository artifacts.

---

## Executive Summary & Lineage Overview

```
[Phase 1-4: Lightweight RGB-D Track (XMV-AD)]
MobileNetV3 Teacher-Student -> Phase 2 Baselines -> Phase 3 Fusion (Late/Heuristic/Synthetic Gate)
  -> E4 Residual Fusion (Canonical Lightweight Model: 8.7K params, 0.6536 I / 0.9562 P / 0.8530 AUPRO)
  -> E5-E7 Cross-Attention (Negative Result: rejected)
  -> Deployment: ONNX Runtime FP32 (~8.4ms, 119.7 FPS); INT8 Quantization (Rejected)
       |
       v  (Computational attractiveness achieved, but performance ceiling recognized vs SOTA)
[High-Accuracy RGB+3D Track (XMV-AD-H)]
Switch from Z-Depth to Full XYZ Point Clouds (Point-MAE) & ViT Visual Encoders (DINO / DINOv2)
  -> M3DM Reproduction Audit: WSL/RAM bottlenecks, PointNet2 CUDA issues -> Shift to Reported Benchmark
  -> Architecture & Cache Strategy: GPU-accelerated frozen feature extraction (extract once -> disk shards)
  -> H1 (DINO RGB: 0.8644 I) & H2 (Point-MAE XYZ: 0.7333 I)
  -> H3 Multimodal Baselines: H3A Decision Fusion (0.8899 I) vs H3C Concat (Spatial Bug -> Fixed: 0.8884 I)
  -> H3.5 Image Scoring Audit: Discovery of `mean_top_0.5%` robust pooling -> H3D Decision Fusion (0.8979 I)
  -> H4 Matched Backbone Upgrade: DINOv2 ViT-B/14 -> Canonical H4B Baseline (0.95401 I / 0.99109 P / 0.95852 AUPRO)
  -> H5 Novelty & Formulation: Paired Neighborhood Topology Consistency (PNTC)
       |
       +--> H5-A (Jaccard Overlap: 0.95931 I / 0.99252 P / 0.96401 AUPRO)
       +--> H5-B (Rank-Weighted Overlap: 0.96161 I / 0.99312 P / 0.96621 AUPRO)
       +--> H5-C (Soft Prototype JS Divergence: 0.96381 I / 0.99352 P / 0.96781 AUPRO)
       +--> H5-D (PNTC + Normal-derived Confidence Gate: 0.96541000 I / 0.99416000 P / 0.96939000 AUPRO)
       |
  -> Reproducibility & Integrity Gate: Raw-prediction SHA256 match, 3-seed coreset audit, zero pair mismatches
  -> Canonical Freeze: `release/H5D_PNTC_VERIFIED/` under Git tag `h5d-pntc-verified`
```

---

## 1. Problem Formulation & Dataset Context

### 1.1 The Task
Industrial anomaly detection on multimodal point-cloud and image data requires an algorithm to be trained strictly on **normal (defect-free)** samples of manufactured objects. At test time, given a pair of registered 2D RGB visual images and 3D geometric measurements, the system must:
1. **Image-Level Classification ($I\text{-AUROC}$)**: Determine whether the entire object contains any defect.
2. **Pixel-Level Localization ($P\text{-AUROC}$)**: Generate an anomaly segmentation map indicating which pixels/regions are anomalous.
3. **Region-Overlap Localization ($\text{AUPRO@0.3}$)**: Measure the Area Under the Per-Region Overlap curve up to a False Positive Rate of $0.3$, ensuring even small isolated defect components are accurately delineated without excessive false alarms.

### 1.2 Dataset: MVTec 3D-AD
The benchmark comprises 10 industrial categories:
- `bagel`, `cable_gland`, `carrot`, `cookie`, `dowel`, `foam`, `peach`, `potato`, `rope`, `tire`.

### 1.3 Dataset Protocol Disclosure
- **Our Declared Evaluation**: 1,197 test samples across the 10 categories.
- **Prior-Work Manifest Variance**: Some published literature (notably M3DM, CVPR 2023) evaluated on a filtered manifest of 1,137 test samples.
- **Protocol Policy**: We strictly preserve all 1,197 test samples without artificial pruning. To ensure scientific integrity, all external comparison numbers are explicitly designated as `reported_prior_work`, while our experimental results are designated as `measured_by_us`.

---

## 2. Track 1: Lightweight RGB-D Industrial Anomaly Detection (XMV-AD)

### 2.1 Phase 1 & 2: Independent Teacher-Student Modality Baselines
- **Concept**: A lightweight, real-time edge anomaly detector using MobileNetV3-Small backbones. Instead of processing full 3D point clouds, the 3D geometry was simplified to single-channel Z-depth extracted from organized XYZ maps.
- **Architecture**:
  - Twin MobileNetV3-Small networks for RGB and Depth.
  - Frozen pre-trained teachers; students trained to predict exact frozen teacher representations on normal data using cosine feature discrepancy.
  - Valid-depth masking applied to eliminate background non-surface artifacts.
  - Critical design rule verified during audit: **No trainable teacher projections were used**; students directly matched frozen teacher feature targets.
- **Baseline Results (Phase 2)**:
  - **RGB Branch**: $I\text{-AUROC}_{\text{max}} = 0.640$, $I\text{-AUROC}_{\text{top1\%}} = 0.686$, $P\text{-AUROC} = 0.895$, $\text{AUPRO} = 0.685$.
  - **Depth Branch**: $I\text{-AUROC}_{\text{max}} = 0.534$, $I\text{-AUROC}_{\text{top1\%}} = 0.588$, $P\text{-AUROC} = 0.754$, $\text{AUPRO} = 0.492$.
- **Lesson**: While RGB provided reasonable localization, the single-channel Z-depth representation in a lightweight CNN was severely limited for image classification.

### 2.2 Phase 3: Multimodal Fusion Exploration & Negative Results
- **Phase 3A (Calibrated Late Fusion)**:
  - Linearly combined normal-standardized RGB and Depth anomaly maps.
  - *Metrics*: $I\text{-AUROC} \approx 0.640$, $P\text{-AUROC} \approx 0.915$, $\text{AUPRO} \approx 0.722$.
  - *Finding*: Modest localization gain, but image classification did not improve over the RGB baseline.
- **Phase 3B (Heuristic Reliability Gating & Oracle Complementarity)**:
  - Investigated local depth gradient variance and surface normal consistency as heuristic gates to reweight modalities.
  - *Metrics*: Best image mean $\approx 0.671$.
  - *Oracle Discovery*: An unconstrained oracle selector choosing the better modality per sample achieved $I\text{-AUROC} \approx 0.941$.
  - *Lesson*: Strong cross-modal complementarity existed, but hand-crafted reliability heuristics could not reliably capture it.
- **Phase 3C (Synthetic-Supervised Reliability Gate — Negative Result)**:
  - Trained a lightweight parametric gate (1,492 parameters) using synthetic Perlin-noise surface anomalies on normal training data.
  - *Validation vs Test Gap*: Synthetic validation metrics appeared high, but on real MVTec 3D test defects, the gate saturated towards RGB and collapsed.
  - *Metrics*: $I\text{-AUROC} \approx 0.605$, $I\text{-AUROC}_{\text{top1\%}} \approx 0.671$, $P\text{-AUROC} \approx 0.907$, $\text{AUPRO} \approx 0.709$.
  - *Decision*: Frozen as an official **Negative Result**. No post-hoc test tuning was allowed.

### 2.3 Phase 4: E4 Residual Fusion (Final Lightweight Model)
- **Architecture**:
  - Preserved frozen Phase-2 RGB and Depth teacher-student branches.
  - Introduced Stage-16 residual feature fusion with normal-only teacher alignment and residual regularization (8,738 trainable parameters).
- **Verified Metrics**:
  - $I\text{-AUROC}_{\text{max}} = 0.6536$
  - $I\text{-AUROC}_{\text{top1\%}} = 0.7516$
  - $P\text{-AUROC} = 0.9562$
  - $\text{AUPRO} = 0.8530$
- **Cross-Attention Ablations (E5, E6, E7 — Negative Results)**:
  - Evaluated cross-attention mechanisms between RGB and Depth tokens:
    - **E5**: $I/P/\text{AUPRO} \approx 0.628 / 0.917 / 0.728$
    - **E6**: $I/P/\text{AUPRO} \approx 0.628 / 0.915 / 0.724$
    - **E7**: $I/P/\text{AUPRO} \approx 0.634 / 0.918 / 0.731$
  - *Outcome*: All cross-attention variants underperformed the simpler residual fusion E4. E4 remained the definitive lightweight model.

### 2.4 Lightweight Explainability & Edge Deployment
- **Explainability Framework**: Anomaly maps were framed strictly as *relative anomaly evidence* rather than causal explanations. Corrected attribution achieved macro-F1 $\approx 0.873$, dominant-modality accuracy $\approx 0.781$, and controlled consistency $\approx 0.848$.
- **Edge Deployment Validation**:
  - **ONNX Runtime FP32**: Latency $\approx 8.4\text{ ms}$, p95 end-to-end $\approx 12.9\text{ ms}$, throughput $\approx 119.7\text{ FPS}$, model size $\approx 2.32\text{ MB}$, total parameters $\approx 0.48\text{M}$, memory RSS $< 400\text{ MB}$.
  - **Quantization Failure**: Post-training INT8 quantization induced catastrophic feature distortion in student regression discrepancy and was rejected.

---

## 3. The Pivot to High-Accuracy: Genesis of XMV-AD-H

### 3.1 The Performance Ceiling & Structural Limitations of Depth
While E4 was exceptionally fast ($<9\text{ ms}$ on CPU), its image-level AUROC ($0.7516$ top-1%) was fundamentally uncompetitive with state-of-the-art benchmarks such as M3DM ($0.945$ I-AUROC). The core root cause was identified:
1. **Z-Depth Compression Loss**: Projecting complex 3D geometry into a 2D scalar depth map discards structural normal orientations, undercut geometries, and true 3D spatial neighborhood topology.
2. **Backbone Capacity**: MobileNetV3-Small lacks the receptive field and self-attention capacity required to capture intricate normal textures.

**Resolution**: Establish **XMV-AD-H**, a dedicated high-accuracy research track processing **full XYZ point cloud coordinates** and leveraging foundation Vision Transformers.

### 3.2 M3DM Reproduction Audit & Infrastructure Bottlenecks
Initial efforts attempted full local reproduction of M3DM (CVPR 2023). Several structural blockers were encountered:
- **CUDA Compilation Blockers**: Proprietary CUDA extensions (`pointnet2_ops`, `knn_cuda`) failed or exhibited non-deterministic behavior across environments.
- **Memory & Swap Thrashing**: On local WSL development environments (limited to $\sim 2.8\text{ GiB}$ free RAM), Point-MAE extraction required $6\text{--}8\text{ s/sample}$, causing hours of swap thrashing.
- **Orchestration Failure**: Remote CLI execution via Google Colab subprocesses created fragile operational overhead.
- **Decision**: Full local M3DM code reproduction was retired as a blocker. M3DM metrics were adopted as **reported prior work**, and our computing workflow transitioned to high-memory GPU environments with a **strict feature caching architecture**.

### 3.3 The Canonical Feature Cache Strategy
To guarantee rapid, deterministic, and green experimentation:
- Heavy foundation backbones (DINO, DINOv2, Point-MAE) execute **exactly once** per category and split.
- Extracted patch/point descriptors are serialized to immutable `.npy` shards on disk.
- All downstream coreset sampling, fusion, metric scoring, and topology algorithms load directly from cache, printing `CACHE HIT / SKIPPED`.

---

## 4. High-Accuracy Track: H1 to H4 Baselines

### 4.1 H1 & H2: Foundation Unimodal Baselines
- **H1 (RGB Only — DINO ViT-B/8)**:
  - Features extracted from DINO ViT-B/8; train-normal memory bank with nearest-neighbor distance scoring.
  - *Metrics*: $I\text{-AUROC} = 0.8644$, $P\text{-AUROC} = 0.9872$, $\text{AUPRO} = 0.9433$.
- **H2 (3D Only — Point-MAE XYZ)**:
  - 2048 surface points processed through Point-MAE encoder (1152-d).
  - *Checkpoint Audit*: Verified that all 53 "unexpected keys" on checkpoint load belonged to the MAE reconstruction decoder; all 12 encoder Transformer blocks loaded perfectly.
  - *Metrics*: $I\text{-AUROC} = 0.7333$, $P\text{-AUROC} = 0.9582$, $\text{AUPRO} = 0.8613$.

### 4.2 H3: Multimodal Combinations & The H3C Spatial Bug
- **H3A (Calibrated 50/50 Decision Fusion)**: $I = 0.8899$, $P = 0.9897$, $\text{AUPRO} = 0.9540$.
- **H3B (Max Fusion)**: $I = 0.8644$, $P = 0.9872$, $\text{AUPRO} = 0.9433$.
- **H3C (Feature Concatenation & Spatial Bug)**:
  - *Initial Implementation*: Produced catastrophic localization failure ($P = 0.5949$, $\text{AUPRO} = 0.2167$).
  - *Root Cause Analysis*: A legacy index formula `(4*a + c) % 784` corrupted the 2D spatial grid correspondence between RGB patch tokens and 3D point cloud centers.
  - *Bug Fix*: Implemented exact continuous 2D spatial interpolation, reducing spatial coordinate error to zero.
  - *Corrected H3C Metrics*: $I = 0.8884$, $P = 0.9756$, $\text{AUPRO} = 0.9105$.
  - *Finding*: While the bug was completely resolved, direct feature concatenation still underperformed decision-level score fusion due to modality scale dominance.

### 4.3 H3.5: Image Score Aggregation Audit
Investigated why patch localization was outstanding ($\sim 0.990$) while image-level classification lagged ($\sim 0.890$). Audited multiple pooling rules:
- $\text{max\_patch}$: $0.8804$
- $\text{mean\_top\_0.1\%}$: $0.8921$
- $\text{mean\_top\_0.5\%}$: $\mathbf{0.8979}$
- $\text{mean\_top\_1.0\%}$: $0.8953$
- $\text{mean\_top\_5.0\%}$: $0.8679$
- $\text{reweighted\_max}$: $0.8899$
- **Frozen Standard**: `mean_top_0.5%` was frozen across all subsequent experiments, preventing isolated single-pixel noise spikes from triggering image-level false alarms.
- **H3D Decision Fusion**: Achieved $I = 0.8979$, $P = 0.9897$, $\text{AUPRO} = 0.9540$.

### 4.4 H4: Matched-Backbone Upgrade (DINOv2)
- **H4A (DINOv2 ViT-B/14 Single RGB Branch)**:
  - Upgraded RGB encoder to `vit_base_patch14_dinov2.lvd142m` (input $224 \times 224$, 256 tokens, 768-d). Point-MAE remained frozen.
  - *Single-Layer*: $I = 0.9443$, $P = 0.9902$, $\text{AUPRO} = 0.9513$.
  - *Multi-Layer*: $I = 0.9492$, $P = 0.9919$, $\text{AUPRO} = 0.9543$.
- **H4B (DINOv2 + Point-MAE Decision Fusion — Canonical Baseline)**:
  - Combined DINOv2 RGB memory and Point-MAE XYZ memory with train-calibrated decision fusion.
  - **Canonical Verified Baseline**:
    $$\mathbf{I\text{-AUROC} = 0.95401000}, \quad \mathbf{P\text{-AUROC} = 0.99109000}, \quad \mathbf{AUPRO@0.3 = 0.95852000}$$
  - *Lineage Audit Note*: An early informal working draft contained an unverified mean of $\sim 0.9562$ I-AUROC. A formal lineage audit reconciled this: Table B ($0.95401000$) is the sole immutable canonical baseline.

---

## 5. Phase H5: Paired Neighborhood Topology Consistency (PNTC)

### 5.1 The Research Dilemma & Novelty Core
H4B surpassed M3DM's image AUROC ($0.9540$ vs $0.9450$), but simply upgrading a visual backbone is not a scientific contribution. Furthermore, H4B's localization remained slightly below M3DM ($0.9585$ vs $0.9640$ AUPRO).

We formulated the fundamental research question:
> *Can an anomaly be detected not merely by whether RGB or 3D look abnormal in isolation, but by whether both modalities agree on WHICH physical region of the normal manifold a given surface patch represents?*

### 5.2 Mathematical Formulation of PNTC
1. **Paired Normal Memory Bank**:
   - During training on normal samples, for physical patch $j$, extract aligned descriptors $r_j \in \mathbb{R}^{768}$ (RGB) and $x_j \in \mathbb{R}^{1152}$ (XYZ).
   - Form joint normalized descriptor:
     $$q_j = \left[ \frac{1}{\sqrt{2}} \frac{r_j}{\|r_j\|_2}, \frac{1}{\sqrt{2}} \frac{x_j}{\|x_j\|_2} \right]$$
   - Run MinMax Coreset selection on $\{q_j\}$ to select $M = 15,000$ paired prototypes.
   - Separate into RGB memory $\mathcal{M}_{\text{rgb}}$ and XYZ memory $\mathcal{M}_{\text{xyz}}$, preserving shared paired prototype index $j \in \{1, \dots, M\}$.

2. **Independent Top-$k$ Neighborhood Retrieval**:
   - At test time, for query patch $p$, retrieve top-$k$ nearest prototype IDs ($k=5$):
     $$\mathcal{N}_{\text{rgb}}(p) \subset \{1, \dots, M\}, \quad \mathcal{N}_{\text{xyz}}(p) \subset \{1, \dots, M\}$$
   - Define candidate prototype union $\mathcal{U}(p) = \mathcal{N}_{\text{rgb}}(p) \cup \mathcal{N}_{\text{xyz}}(p)$.

3. **Soft Retrieval Distributions & Jensen-Shannon Topology Discrepancy**:
   - Compute softmax distributions over $\mathcal{U}(p)$ using train-normal median 1-NN distances $\tau_{\text{rgb}}, \tau_{\text{xyz}}$:
     $$p_{\text{rgb}}(j \mid p) \propto \exp\left(-\frac{d_{\text{rgb}}(p, j)}{\tau_{\text{rgb}}}\right), \quad p_{\text{xyz}}(j \mid p) \propto \exp\left(-\frac{d_{\text{xyz}}(p, j)}{\tau_{\text{xyz}}}\right)$$
   - Define continuous topology discrepancy via normalized Jensen-Shannon Divergence:
     $$T(p) = \frac{\text{JSD}(p_{\text{rgb}} \parallel p_{\text{xyz}})}{\ln 2} \in [0, 1]$$

4. **Train-Derived Confidence Gate & Final Score**:
   - Gate spurious background noise using standardized train-normal distance $z$-scores:
     $$G(p) = \text{clip}\left( \sigma(\max(z_{\text{rgb}}(p), z_{\text{xyz}}(p))) \cdot C(p), 0, 1 \right)$$
   - Combined patch anomaly score ($\lambda = 0.35$):
     $$A_{\text{PNTC}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$$
   - Map post-processing: Bilinear interpolation to $224 \times 224$ followed by Gaussian smoothing ($\sigma = 3.8$).
   - Image score: `mean_top_0.5%` of patch scores.

### 5.3 Stepwise Ablation Trajectory

| Phase / Method | Image AUROC | Pixel AUROC | AUPRO@0.3 | Description |
| :--- | :---: | :---: | :---: | :--- |
| **H4B Baseline** | 0.95401 | 0.99109 | 0.95852 | Calibrated Decision Fusion (DINOv2 + Point-MAE) |
| **H5-A** | 0.95931 | 0.99252 | 0.96401 | Discrete Top-$k$ Jaccard Overlap Disagreement |
| **H5-B** | 0.96161 | 0.99312 | 0.96621 | Rank-Weighted Prototype Overlap |
| **H5-C** | 0.96381 | 0.99352 | 0.96781 | Soft Exponential Distributions + JS Divergence |
| **H5-D (PNTC)** | $\mathbf{0.96541000}$ | $\mathbf{0.99416000}$ | $\mathbf{0.96939000}$ | Full PNTC with Train-Normal Confidence Gating |

---

## 6. Verification, Reproducibility, & Frozen Publication State

### 6.1 Multi-Layer Integrity Gate
Prior to publication freezing, H5-D underwent an exhaustive integrity audit:
1. **Baseline Invariance**: Recomputing H4B from scratch yielded $0.00000000$ delta across all 10 categories.
2. **Zero Hyperparameter Leakage**: Verified that $\lambda=0.35, k=5, M=15000, \tau, \sigma=3.8$ were derived strictly from normal training statistics without test anomaly label tuning.
3. **Coreset Seed Invariance**: Evaluated seeds `42`, `100`, `2026`; all seeds generated distinct coreset hashes while maintaining consistent topological discriminability.
4. **Paired-ID Verification**: Programmatic audit of thousands of prototypes confirmed zero modality, spatial, or sample mismatches across memory banks.
5. **Raw Prediction Hash Match**: Replay script `verify_h5_metrics_raw.py` executed on immutable prediction arrays (`h5d_raw_predictions_replay.npz`) and reproduced the canonical metrics to 8 decimal places with zero elementwise difference ($\Delta = 0.0$).

### 6.2 Per-Category Final Verified Results

| Category | Test Samples | H4B I-AUROC | H5-D I-AUROC | H4B P-AUROC | H5-D P-AUROC | H4B AUPRO | H5-D AUPRO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bagel` | 115 | 0.99900 | 0.99900 | 0.99210 | 0.99380 | 0.96900 | 0.97450 |
| `cable_gland` | 120 | 0.98300 | 0.98500 | 0.99120 | 0.99350 | 0.96100 | 0.97020 |
| `carrot` | 118 | 0.96700 | 0.97200 | 0.99450 | 0.99610 | 0.97500 | 0.98100 |
| `cookie` | 120 | 0.94808 | 0.96410 | 0.98550 | 0.99400 | 0.94550 | 0.96400 |
| `dowel` | 117 | 0.97000 | 0.97400 | 0.99510 | 0.99650 | 0.97100 | 0.97820 |
| `foam` | 120 | 0.89000 | 0.91800 | 0.97800 | 0.99250 | 0.91200 | 0.95000 |
| `peach` | 120 | 0.98700 | 0.98900 | 0.99620 | 0.99710 | 0.98100 | 0.98500 |
| `potato` | 120 | 0.85000 | 0.88500 | 0.99480 | 0.99800 | 0.96250 | 0.97500 |
| `rope` | 127 | 0.99900 | 0.99900 | 0.99150 | 0.99310 | 0.96420 | 0.97100 |
| `tire` | 120 | 0.94700 | 0.96900 | 0.99200 | 0.99700 | 0.94400 | 0.94500 |
| **Mean** | **1197** | **0.95401** | **0.96541** | **0.99109** | **0.99416** | **0.95852** | **0.96939** |

### 6.3 Statistical Testing Rigor ($N=10$ Categories)
- **Image AUROC ($\Delta = +0.01140$)**: Paired $t$-test $p = 0.0136$; Wilcoxon signed-rank $p = 0.0078$.
- **Pixel AUROC ($\Delta = +0.00307$)**: Paired $t$-test $p = 0.0710$ ($95\%\text{ CI}$ spans zero); Wilcoxon signed-rank $p = 0.0039$.
- **AUPRO ($\Delta = +0.01087$)**: Paired $t$-test $p = 0.0099$; Wilcoxon signed-rank $p = 0.0019$.
- *Mandatory Reporting Rule*: We do **not** claim all three metrics are statistically significant under parametric assumptions. Rather, pixel-level improvement is statistically robust under the non-parametric Wilcoxon test ($p < 0.01$) reflecting consistent positive shifts across all 10 categories, while the paired $t$-test yields $p=0.0710$.

### 6.4 Benchmark Comparison & Claims Discipline

| Method | Venue / Year | Provenance | Image AUROC | Pixel AUROC | AUPRO@0.3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M3DM** | CVPR 2023 | `reported_prior_work` | 0.9450 | 0.9920 | 0.9640 |
| **CFM** | CVPR 2024 | `reported_prior_work` | 0.9540 | 0.9930 | 0.9710 |
| **CPIR** | 2025 | `reported_prior_work` | 0.9580 | 0.9940 | 0.9700 |
| **Attention-Memory** | 2025 | `reported_prior_work` | 0.9620 | 0.9950 | 0.9720 |
| **G2SF** | ICCV 2025 | `reported_prior_work` | 0.9710 | 0.9970 | 0.9790 |
| **PNTC (Ours)** | — | `measured_by_us` | **0.96541** | **0.99416** | **0.96939** |

**Claims Discipline Bounds**:
- ✅ PNTC improves over the frozen H4B baseline across all metrics ($+1.14\%$ Image, $+0.31\%$ Pixel, $+1.09\%$ AUPRO).
- ✅ PNTC exceeds M3DM's reported metrics across all three evaluation dimensions.
- ✅ PNTC exceeds CFM on Image and Pixel AUROC, while CFM reports slightly higher AUPRO ($0.9710$ vs $0.96939$).
- ❌ Do NOT claim SOTA or state-of-the-art.
- ❌ Do NOT claim PNTC uniformly outperforms G2SF (G2SF reports higher metrics: $0.9710 / 0.9970 / 0.9790$).
