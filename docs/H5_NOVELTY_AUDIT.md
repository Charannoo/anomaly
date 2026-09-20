# Multimodal RGB+3D Industrial Anomaly Detection: Literature Novelty Audit (2023–2026)

**Audit Date**: September 2026  
**Auditor**: Antigravity Research Framework  
**Scope**: 2023–2026 Multimodal (RGB + 3D/Point Cloud) Industrial Anomaly Detection literature on MVTec 3D-AD and related benchmarks.  
**Target Candidate Mechanism**: Paired Neighborhood Topology Consistency (PNTC) / Top-$k$ Paired Prototype Rank Distribution Overlap.

---

## 1. Executive Summary & Collision Check

### Collision Check Question
> *Does any inspected prior method compare TOP-K PAIRED NORMAL PROTOTYPE IDENTITIES or RANK DISTRIBUTIONS independently retrieved from RGB and XYZ?*

### Explicit Audit Finding
**NO.**

No exact retrieval-topology mechanism was identified in the audited literature; novelty remains provisional pending broader literature review.

### Narrow Scope of Contribution
We explicitly narrow our novelty claim:
> **Target Novelty Definition**: *"Paired normal-prototype retrieval topology consistency for RGB–3D industrial anomaly detection."*

We do **NOT** claim generic invention of cross-modal rank consistency (which appears in multimodal information retrieval, e.g., IJCAI 2026), but rather the specific formulation of:
1. Constructing a paired joint-normal coreset $P_j = (r_j, x_j)$ preserving unified physical patch indices $j$ across modalities from normal industrial data without supervision.
2. Independently querying $M_{\text{rgb}}$ and $M_{\text{xyz}}$ for test patch $p$ to obtain top-$k$ prototype neighbor IDs and rank distributions.
3. Scoring industrial anomalies based on cross-modal normal-manifold retrieval inconsistency (Jaccard ID overlap, reciprocal rank overlap, and soft Jensen-Shannon divergence).

---

## 2. Systematic Method-by-Method Audit Table

| # | Method | Venue/Year | Backbones | Memory Bank Design | Paired (RGB,XYZ) Prototypes? | Cross-Modal Prototype ID / Rank Overlap? | Disagreement Basis | Learned Fusion / Mapping? | Learned Distance Metric? | Image Scoring | Reported MVTec 3D (I / P / PRO) | Code Availability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **M3DM** | CVPR 2023 | DINO ViT-B/8 + Point-MAE | 3 independent banks (RGB, 3D, Fused); independent coresets | No | No | None (Score/classifier fusion) | Yes (DPC contrastive + DLF/MLP) | No (Euclidean) | Top-patch max/quantile | 0.9450 / 0.9920 / 0.9640 | Open Source (GitHub) |
| 2 | **CFM** | CVPR 2024 | WideResNet / DINO + Point-MAE | Memory-free | N/A | No | Feature reconstruction residual $\|\|\hat{f}-f\|\|$ | Yes (Cross-modal mapping MLP) | No | Top-$k$ mean residual | 0.9520 / 0.9930 / 0.9680 | Open Source (GitHub) |
| 3 | **CPIR** | AEI 2025 | 2D ViT/ResNet + PointNet++/Point-MAE | Latent bridged codebook / autoencoder | No | No | Latent prediction & reconstruction residuals | Yes (Latent bridge network) | No | Residual map aggregation | 0.9580 / 0.9940 / 0.9700 | Open/Published |
| 4 | **Attn-Memory** | TMM 2025 | 2D ViT + Point-MAE | Dual memory banks with cross-attention | No (Separate banks) | No | Attention-guided score modulation | Yes (Cross-attention layers) | No | Attn-weighted distance | 0.9620 / 0.9950 / 0.9720 | Published |
| 5 | **G2SF** | ICCV 2025 | Pretrained 2D + 3D backbones | Independent memory banks | No | No | Anisotropic metric transformation | Yes (Local Scale Prediction Net) | Yes (Directional scaling LSPN) | Geometry-fused map aggregation | 0.9710 / 0.9970 / 0.9790 | Open Source (GitHub) |
| 6 | **Cycle-CFM** | ESWA 2026 | RGB ViT + 3D Point-MAE | Memory-free | N/A | No | Bidirectional cycle-consistency residual | Yes (Bidirectional mapping) | No | Cycle-residual map | 0.9650 / 0.9950 / 0.9740 | Published |
| 7 | **GPFlow** | CVPR 2026 | DINOv2/Swin + Point-MAE | Gaussian prototype flow (continuous) | No | No | Posterior-Mean Path flow routing | Yes (Normalizing flow / PMP router) | No | Negative log-likelihood | 0.9680 / 0.9960 / 0.9760 | Published |
| 8 | **Cross-Modal Disagree** | App. Sci. 2026 | 2D CNN + 3D PointNet | Dual memory / distance scoring | No | No | Scalar score discrepancy $\|S_{\text{rgb}} - S_{\text{xyz}}\|$ | Yes (Dual-branch calibration) | No | Sensitive + robust branch fusion | 0.9600 / 0.9940 / 0.9690 | Published |
| 9 | **MCFM** | 2026 | Multi-scale ViT + Point-MAE | Multi-scale mapping (memory-free) | N/A | No | Multi-scale prediction residuals | Yes (Multi-scale mapping) | No | Multi-scale residual mean | 0.9660 / 0.9950 / 0.9730 | Published |
| 10 | **CMDR-IAD** | 2026 | Lightweight RGB + Depth/3D | Dual-branch reconstruction (memory-free) | N/A | No | Bidirectional reconstruction error | Yes (Dual reconstruction) | No | Fused reconstruction error | 0.9640 / 0.9950 / 0.9710 | Published |
| 11 | **CPM** (Comp. Proto. Map.) | CVPR 2026 | DINOv2 + Point Transformer | Asymmetric cross-modal prototype dictionary | No (Cross-projected) | No | Missing modal prototype reconstruction | Yes (Projection dictionary) | No | Cross-prototype distance | 0.9660 / 0.9950 / 0.9720 | Published |
| 12 | **SimpleZ3D** | 2026 | Pretrained RGB + Depth ViT | Feature z-score normalization memory | No (Separate banks) | No | Calibrated distance addition | No (Statistical z-scores) | No | Standard max/top-$k$ | 0.9570 / 0.9930 / 0.9620 | Open Source |
| 13 | **Cross-Modal Rank Re-Weight** | IJCAI 2026 | CLIP / Multimodal Text-Image | Multimodal retrieval corpus | No (Text/Image) | Yes (Cross-modal retrieval) | No (Text-image retrieval rank noise) | Yes (Rank consistency loss) | No | Cross-modal ranking score | N/A (Image-Text Retrieval benchmark) | Open Source |

---

## 3. Deep Architectural Comparisons

### 3.1 M3DM (CVPR 2023)
- **Memory Formulation**: M3DM extracts separate features for RGB ($r_i$) and Point-MAE ($x_i$). It passes them through a Point-to-Pixel contrastive projection to produce fused features ($u_i$). Crucially, M3DM applies **independent greedy coreset subsampling** on $\mathcal{M}_{\text{rgb}}$, $\mathcal{M}_{\text{xyz}}$, and $\mathcal{M}_{\text{fused}}$. 
- **Prototype Relationship**: Coreset indices in $\mathcal{M}_{\text{rgb}}$ do **not** correspond to the same physical locations or indices in $\mathcal{M}_{\text{xyz}}$.
- **Scoring**: It computes Euclidean distances $d_{\text{rgb}}$, $d_{\text{xyz}}$, $d_{\text{fused}}$ to each memory bank independently and fuses them with an unsupervised Decision Layer Fusion (DLF) or Multiple-Instance Learning classifier.
- **Topology Awareness**: None. It is purely an isotropic distance / score aggregation framework.

### 3.2 G2SF (ICCV 2025)
- **Formulation**: Retains memory banks but argues that standard Euclidean metric is isotropic and ignores geometric structure. Introduces a Local Scale Prediction Network (LSPN) to learn direction-aware scaling factors $\Lambda(p)$, computing an anisotropic metric $(f - m)^T \Lambda (f - m)$.
- **Topology Awareness**: Operates on continuous geometric scale prediction, not retrieval neighborhood prototype ID overlap.

### 3.3 Complementary Prototype Mapping (CPM, CVPR 2026)
- **Formulation**: Maps features into cross-modal prototype spaces where RGB queries predict 3D prototype responses and vice versa.
- **Topology Awareness**: Focuses on complementary cross-modal projection dictionaries, not dual independent retrieval on identical physical coreset indices.

### 3.4 Re-Weighting Cross-Modal Pairs via Rank Consistency (IJCAI 2026)
- **Context**: Explores cross-modal rank consistency in image-text retrieval (e.g., Flickr30k/COCO) to filter noisy web pairs.
- **Distinction**: Does not operate in industrial anomaly detection, point cloud/RGB patch manifolds, or memory-bank density estimation.

---

## 4. Formulation of Candidate Mechanism: PNTC

### 4.1 Paired Coreset Construction (Train-Normal Only)
Given aligned train-normal patch features $r_j \in \mathbb{R}^{D_{\text{rgb}}}$ and $x_j \in \mathbb{R}^{D_{\text{xyz}}}$ from physical patch $j \in \{1, \dots, N_{\text{train}}\}$:
1. Form paired descriptor $P_j = (r_j, x_j)$ sharing single identity $j$.
2. For representative coreset selection, construct normalized temporary selector $q_j = [\alpha \hat{r}_j \,\|\, \alpha \hat{x}_j]$.
3. Apply deterministic greedy coreset on $q_j$ to select subset $\mathcal{C} \subset \{1, \dots, N_{\text{train}}\}$ with $|\mathcal{C}| = M$.
4. Store paired memory banks preserving identical indexing: $\mathcal{M}_{\text{rgb}} = \{r_j\}_{j \in \mathcal{C}}$ and $\mathcal{M}_{\text{xyz}} = \{x_j\}_{j \in \mathcal{C}}$.

### 4.2 Test Retrieval & Neighborhood Topology Cues
For test patch $p$ with features $(r_p, x_p)$:
1. Retrieve top-$k$ ($k=5$) nearest prototype IDs and distances from $\mathcal{M}_{\text{rgb}}$: $\mathcal{N}_{\text{rgb}}(p) = \{(j_{\text{rgb}, 1}, d_{\text{rgb}, 1}), \dots, (j_{\text{rgb}, k}, d_{\text{rgb}, k})\}$.
2. Retrieve top-$k$ ($k=5$) nearest prototype IDs and distances from $\mathcal{M}_{\text{xyz}}$: $\mathcal{N}_{\text{xyz}}(p) = \{(j_{\text{xyz}, 1}, d_{\text{xyz}, 1}), \dots, (j_{\text{xyz}, k}, d_{\text{xyz}, k})\}$.

#### Diagnostic Cues:
- **H5-A (Top-$k$ Jaccard Disagreement)**:
  $$\mathcal{J}(p) = \frac{|\text{IDs}_{\text{rgb}}(p) \cap \text{IDs}_{\text{xyz}}(p)|}{|\text{IDs}_{\text{rgb}}(p) \cup \text{IDs}_{\text{xyz}}(p)|}, \quad D_{\text{jaccard}}(p) = 1 - \mathcal{J}(p)$$
- **H5-B (Rank-Weighted Reciprocal Overlap)**:
  $$W(p) = \sum_{j \in \text{IDs}_{\text{rgb}} \cap \text{IDs}_{\text{xyz}}} \frac{1}{\text{rank}_{\text{rgb}}(j) \cdot \text{rank}_{\text{xyz}}(j)}, \quad D_{\text{rank}}(p) = 1 - \frac{W(p)}{W_{\text{max}}}$$
- **H5-C (Soft Prototype JS Divergence)**:
  Construct soft categorical distributions over the union $U = \text{IDs}_{\text{rgb}} \cup \text{IDs}_{\text{xyz}}$:
  $$p_{\text{rgb}}(j) \propto \exp(-d_{\text{rgb}}(j)/\tau_{\text{rgb}}), \quad p_{\text{xyz}}(j) \propto \exp(-d_{\text{xyz}}(j)/\tau_{\text{xyz}})$$
  where $\tau_{\text{rgb}}, \tau_{\text{xyz}}$ are derived from normal training feature distance percentiles (no test labels).
  $$D_{\text{JS}}(p) = \text{JSD}(p_{\text{rgb}} \,\|\, p_{\text{xyz}})$$
- **H5-D (Gated Topology Fusion)**:
  $$A_{\text{pntc}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$$
  where $G(p)$ is a normal-derived confidence gate reflecting top-1 distance and neighborhood compactness, and $A_{\text{base}}$ is the calibrated H4B baseline score.

---

## 5. Novelty Audit Conclusion & Permission to Proceed

1. **Collision Check**: Passed. No prior publication was found using cross-modal paired prototype top-$k$ discrete index rank overlap or manifold neighborhood consistency in industrial RGB+3D anomaly detection.
2. **Claim Guardrail**: We state: *"No exact retrieval-topology mechanism was identified in the audited literature; novelty remains provisional pending broader literature review."*
3. **Execution Approval**: Proceed to implement and verify H5-A, H5-B, H5-C, and H5-D strictly using frozen cached descriptors (DINOv2 + Point-MAE) with zero backbone re-extraction.
