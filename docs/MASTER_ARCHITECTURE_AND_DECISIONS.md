# Master Architecture and Design Decisions: Architectural Rationale & Failure Analysis

This document provides a comprehensive, rigorous technical justification for every major architectural component and hyperparameter choice in the **Paired Neighborhood Topology Consistency (PNTC)** framework, alongside an in-depth failure analysis of all rejected alternatives.

---

## 1. Architectural Blueprint & Data Flow

```
================================================================================================
                                   OFFLINE TRAINING PHASE (NORMAL SAMPLES ONLY)
================================================================================================

 [Normal RGB Image (224x224)]                             [Normal 3D Point Cloud (2048 pts)]
              |                                                           |
              v                                                           v
  [DINOv2 ViT-B/14 Encoder]                                   [Point-MAE 3D Encoder]
              |                                                           |
              v                                                           v
     Patch Descriptors                                           Point Descriptors
    r_j in R^768 (256 tokens)                                   x_j in R^1152 (256 pts)
              \                                                           /
               \                                                         /
                +---------------> Aligned Physical Patch <---------------+
                                              |
                                              v
                              Joint Normalized Descriptor:
                       q_j = [ 1/sqrt(2) * (r_j / ||r_j||), 
                               1/sqrt(2) * (x_j / ||x_j||) ]
                                              |
                                              v
                               [MinMax Paired Coreset Selection]
                                           (M = 15,000)
                                              |
                                              v
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
            [Paired RGB Memory Bank]                  [Paired XYZ Memory Bank]
               M_rgb in R^(15000 x 768)                  M_xyz in R^(15000 x 1152)
                         |                                         |
                         +------------ Shared Index j -------------+

================================================================================================
                                   ONLINE INFERENCE & ANOMALY SCORING
================================================================================================

  [Test RGB Patch r(p)]                                     [Test 3D Patch x(p)]
            |                                                         |
            v                                                         v
   Query M_rgb (Exact 1-NN & k-NN)                          Query M_xyz (Exact 1-NN & k-NN)
            |                                                         |
            +------------------------+       +------------------------+
            |                        |       |                        |
            v                        v       v                        v
      [Min 1-NN Dist d_rgb]      [Top-k IDs: N_rgb(p)]   [Top-k IDs: N_xyz(p)]      [Min 1-NN Dist d_xyz]
            |                        (k = 5)                 (k = 5)                        |
            v                                \               /                              v
    Train-Standardized                        \             /                       Train-Standardized
      z_rgb(p)                                 \           /                          z_xyz(p)
            \                                   \         /                                 /
             \                                   v       v                                 /
              \                         [Candidate Union U(p)]                            /
               \                                   |                                     /
                \                                  v                                    /
                 \                      [Soft Prototype Probs]                         /
                  \                     p_rgb(j|p), p_xyz(j|p)                        /
                   \                               |                                 /
                    \                              v                                /
                     \                  [Jensen-Shannon Divergence]                /
                      \                           T(p)                            /
                       \                           |                             /
                        \                          v                            /
                         \              [Confidence Gate G(p)]                 /
                          \              clip(sigma(max(z))*C, 0, 1)          /
                           \                       |                         /
                            \                      v                        /
                             +------> [Topology Term: lambda * G(p) * T(p)] <+
                                                   |
                                                   v
                                [Final Patch Score A_PNTC(p)]
                                 = A_base(p) + lambda * G(p) * T(p)
                                                   |
                                                   v
                                 [2D Anomaly Map Interpolation]
                                          (224 x 224)
                                                   |
                                                   v
                                    [Gaussian Smoothing (sigma=3.8)]
                                                   |
                             +---------------------+---------------------+
                             |                                           |
                             v                                           v
                   [Pixel-Level Outputs]                       [Image-Level Aggregation]
                  P-AUROC & AUPRO@0.3 Eval                    mean_top_0.5% Patch Scores
                                                                         |
                                                                         v
                                                                 I-AUROC Evaluation
================================================================================================
```

---

## 2. Deep-Dive: Justification of Core Architectural Decisions

### 2.1 Full 3D XYZ Geometry vs 2D Scalar Depth
- **Decision**: Process unordered 3D point clouds $(X, Y, Z)$ using Point-MAE rather than single-channel 2D depth maps $Z(u,v)$.
- **Rationale**:
  1. *Geometric Invariance*: 2D depth maps suffer from viewpoint sensitivity, self-occlusion discontinuities, and perspective distortion. Point clouds preserve true Euclidean distances and local surface curvature in physical units ($\text{mm}$).
  2. *Structural Defects*: Defects such as structural warpage, surface denting, or thin crack depressions manifest as localized 3D surface normal deviations that disappear under 2D rasterization.
- **Evidence from Project History**:
  - Track 1 (MobileNetV3 Depth): $I\text{-AUROC} = 0.5880$, $P\text{-AUROC} = 0.7540$.
  - Track 2 (Point-MAE XYZ H2): $I\text{-AUROC} = 0.7333$, $P\text{-AUROC} = 0.9582$. Full XYZ alone outperformed 2D depth by $+14.5\%$ Image AUROC and $+20.4\%$ Pixel AUROC.

### 2.2 Visual Backbone: DINOv2 ViT-B/14 vs DINO ViT-B/8 vs CNNs
- **Decision**: Select `vit_base_patch14_dinov2.lvd142m` (pre-trained with self-supervised DINOv2 on 142M images) as the frozen visual feature extractor.
- **Rationale**:
  1. *Dense Semantic Features*: Unlike supervised ImageNet backbones that optimize solely for global category invariance, self-supervised DINOv2 produces extremely fine-grained, patch-level semantic and textural representations ideal for dense anomaly detection.
  2. *Scale and Patch Granularity*: ViT-B/14 transforms a $224 \times 224$ image into a $16 \times 16 = 256$ grid of 768-dimensional token descriptors, matching the 256 geometric centroids produced by Point-MAE.
- **Evidence from Project History**:
  - DINO ViT-B/8 (H1 Baseline): $I\text{-AUROC} = 0.8644$, $P\text{-AUROC} = 0.9872$.
  - DINOv2 ViT-B/14 (H4A Baseline): $I\text{-AUROC} = 0.9443$, $P\text{-AUROC} = 0.9902$.
  - An absolute gain of $+7.99\%$ Image AUROC confirmed DINOv2 as the superior visual foundation.

### 2.3 3D Backbone: Point-MAE vs PointNet++ / DGCNN
- **Decision**: Adopt pre-trained Point-MAE with 12 Transformer encoder blocks operating on 2048 downsampled surface points.
- **Rationale**:
  1. *Self-Supervised Geometric Pre-training*: Point-MAE is pre-trained on ShapeNet via masked point reconstruction, learning rich contextual geometric priors rather than narrow supervised classification features.
  2. *Transformer Alignment*: The Transformer architecture naturally interfaces with Vision Transformers, allowing compatible patch/neighborhood tokenization.
- **Audit Confirmation**: Verified that all 53 unexpected checkpoint keys belonged exclusively to the reconstruction decoder; the 12-layer Transformer encoder loaded with $100\%$ parameter integrity.

### 2.4 Non-Parametric Memory Bank vs Generative Reconstruction
- **Decision**: Utilize non-parametric nearest-neighbor memory banks built strictly on normal prototypes rather than parametric reconstruction (Autoencoders, Normalizing Flows, or GANs).
- **Rationale**:
  1. *Zero "Identical Shortcut" Risk*: Autoencoders frequently suffer from the "identical shortcut" problem, where anomalous patterns (especially subtle color shifts or scratches) are accurately reconstructed by high-capacity decoders, suppressing anomaly signals.
  2. *Deterministic Stability*: Memory banks require zero gradient training iterations on target objects, completely eliminating training divergence, mode collapse, and seed sensitivity.

### 2.5 Paired Normal-Prototype Memory ($P_j = (r_j, x_j)$)
- **Decision**: Enforce a strict 1-to-1 paired indexing scheme where prototype $j$ represents the *identical physical surface patch* across both visual and geometric feature spaces.
- **Rationale**:
  - If RGB and XYZ maintained independent, unlinked memory banks, cross-modal retrieval could only compare distance magnitudes, not topological manifold agreement. Paired indexing allows test patches to query whether visual and geometric neighborhoods map to the *same structural identity* on normal objects.
- **Audit Confirmation**: Programmatic audit (`results/H5_pair_integrity.json`) inspected thousands of prototype pairs across all 10 categories, verifying $0$ spatial, sample, or modality index misalignments.

### 2.6 Independent Modality Retrieval ($k=5$)
- **Decision**: Retrieve top-$k$ prototypes independently in RGB space and XYZ space using each modality's native metric.
- **Rationale**:
  - Joint retrieval (e.g. searching concatenated $[r, x]$) forces an artificial distance metric compromise, allowing one high-magnitude modality to mask discrepancies in the other. Independent retrieval permits each modality to express its purest geometric/visual neighborhood hypothesis.
  - $k=5$ provides sufficient neighborhood context for distributional comparison without diluting the prototype mass into non-local regions.

### 2.7 Continuous Jensen-Shannon Divergence ($T(p)$) vs Discrete Overlap
- **Decision**: Measure neighborhood discrepancy via Jensen-Shannon Divergence over temperature-scaled softmax distributions $p_{\text{rgb}}(j \mid p)$ and $p_{\text{xyz}}(j \mid p)$.
- **Rationale**:
  1. *Symmetry & Boundedness*: JSD is mathematically symmetric ($\text{JSD}(P \parallel Q) = \text{JSD}(Q \parallel P)$) and strictly bounded in $[0, \ln 2]$, allowing clean normalization: $T(p) = \text{JSD} / \ln 2 \in [0, 1]$.
  2. *Smooth Gradients*: Unlike discrete Jaccard distance ($1 - |A \cap B| / |A \cup B|$) which creates step-function artifacts, soft exponential weighting gracefully handles near-miss rankings and distance differences.
- **Ablation Evidence**:
  - Discrete Jaccard (H5-A): $I\text{-AUROC} = 0.95931$.
  - Soft JS Divergence (H5-C): $I\text{-AUROC} = 0.96381$ ($+0.45\%$ improvement).

### 2.8 Train-Derived Confidence Gate ($G(p)$)
- **Decision**: Modulate topology discrepancy $T(p)$ by a confidence gate $G(p) = \text{clip}(\sigma(\max(z_{\text{rgb}}, z_{\text{xyz}})) \cdot C(p), 0, 1)$, parameterized entirely by train-normal distance $z$-scores.
- **Rationale**:
  - In completely normal background regions, slight prototype rank jitter can produce non-zero JSD noise. The confidence gate dampens topology signals in regions where both modalities exhibit low anomaly $z$-scores ($z \ll 0$), activating $T(p)$ only when at least one modality detects ambiguous or suspicious surface features.
- **Integrity Rule**: All normalization statistics ($\mu, \sigma, \tau$) are derived exclusively from normal training samples. Zero test labels or test statistics are used.

### 2.9 Image Anomaly Aggregation: `mean_top_0.5%`
- **Decision**: Aggregate 2D patch anomaly maps into a scalar image anomaly score by taking the mean of the top $0.5\%$ highest-scoring patches.
- **Rationale**:
  - Single maximum pooling ($\text{max\_patch}$) is hypersensitive to isolated camera noise, single-point sensor dropouts, and edge interpolation spikes.
  - Tail averaging over top $0.5\%$ ($\sim 2\text{--}3$ patches in a $256$-patch grid) captures the true spatial extent of compact physical defects while remaining completely robust to isolated random outliers.
- **Audit Evidence**:
  - $\text{max\_patch}$: $I\text{-AUROC} = 0.8804$.
  - $\text{mean\_top\_0.5\%}$: $I\text{-AUROC} = \mathbf{0.8979}$ ($+1.75\%$ gain on H3.5 baseline).

### 2.10 MinMax Coreset Sampling ($M=15,000$)
- **Decision**: Subsample normal training patches using MinMax greedy facility location coreset selection on joint descriptors $q_j$ to a fixed budget $M = 15,000$.
- **Rationale**:
  - Reduces inference nearest-neighbor search complexity by $>90\%$ while preserving maximum coverage of the normal manifold boundary.
  - Seed invariance audits across seeds `42`, `100`, `2026` confirmed invariant metric performance.

---

## 3. Rejected Alternatives & In-Depth Failure Analysis

| Candidate Architecture / Approach | Phase | Experimental Result | Root Cause of Failure / Rejection |
| :--- | :---: | :--- | :--- |
| **Synthetic-Supervised Gating (Phase 3C)** | Track 1 | $I\text{-AUROC} = 0.6050$ (Degraded from $0.6710$) | **Domain Mismatch & RGB Saturation**: Synthetic Perlin noise anomalies on training depth did not mirror real-world physical manufacturing defects (cracks, contamination, deformation). The parametric gate overfitted to synthetic textures and saturated towards RGB during real evaluation. |
| **Cross-Attention Fusion (E5, E6, E7)** | Track 1 | $I\text{-AUROC} \approx 0.628\text{--}0.634$ (Inferior to E4: $0.6536$) | **Over-parameterization on Normal Data**: Training cross-attention modules solely on normal samples forced attention weights to memorize static cross-modal correlation patterns. On anomalous test inputs, attention distributions failed to disperse, suppressing anomaly discrepancies. |
| **Post-Training INT8 Quantization** | Track 1 | Metric collapse ($>5\%$ degradation) | **Cosine Tail Distortion**: Feature discrepancy anomaly detection relies on fine-grained precision in the extreme tails of cosine distance distributions. 8-bit integer quantization destroyed tail separation between subtle anomalies and normal variations. |
| **Direct Feature Concatenation (H3C)** | Track 2 | $P\text{-AUROC} = 0.5949$, $\text{AUPRO} = 0.2167$ (Pre-fix) | **Spatial Grid Desynchronization**: A legacy index formula `(4*a+c)%784` scrambled 2D spatial correspondences between ViT patch tokens and point cloud coordinates. Even after continuous 2D interpolation fixed the bug ($P=0.9756$), concatenation still underperformed decision fusion due to dimensional scale dominance ($768$ vs $1152$). |
| **Elementwise Maximum Fusion (H3B)** | Track 2 | $I\text{-AUROC} = 0.8644$ (Identical to H1 RGB) | **Modality Masking**: Taking the elementwise maximum $\max(S_{\text{rgb}}, S_{\text{xyz}})$ allowed visual noise spikes to consistently dominate subtle geometric anomalies, completely discarding complementary 3D signals. |
| **Multi-Layer DINOv2 Concatenation (H4A Multi)** | Track 2 | $I\text{-AUROC} = 0.9492$ (Marginal gain over single: $0.9443$) | **Dimensionality vs Latency Tradeoff**: Concatenating intermediate ViT layers $[8, 9, 10, 11]$ quadrupled the descriptor dimension ($3072$-d), increasing memory footprint and nearest-neighbor search latency by $400\%$ for a modest exploratory gain. Single final-layer extraction was retained for efficiency and elegance. |
| **Local Full M3DM Code Reproduction** | Track 2 | Pipeline execution stalled ($>6\text{ s/sample}$) | **Environment & CUDA Incompatibilities**: Proprietary `pointnet2_ops` compilation issues and WSL swap thrashing on CPU/low-RAM instances rendered full local execution a massive distraction. Pragmatically replaced with reported prior-work comparison and strict feature caching. |

---

## 4. Efficiency & Computational Complexity Profile

| Metric / Resource | Baseline (H4B) | Final PNTC (H5-D) | Marginal Delta | Rationale / Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Trainable Parameters** | 0 | 0 | **+0** | Non-parametric prototype consistency. Zero gradient updates. |
| **Persistent Parameter Memory** | Negligible | Negligible | **+0 MB** | Only scalar parameters ($\lambda, \tau, \mu, \sigma$) stored. |
| **CPU RAM RSS Overhead** | 382.4 MB | 396.6 MB | **+14.2 MB** | Small temporary buffer for top-$k$ union and JSD calculations. |
| **GPU VRAM Overhead** | 0.0 MB | 0.0 MB | **+0.0 MB** | Memory-bank retrieval executed on CPU/Host memory; zero GPU overhead. |
| **Inference Latency Delta** | 18.2 ms/sample | 22.4 ms/sample | **+4.2 ms/sample** | Fast vectorized JSD computation over small $k=5$ candidate sets. |

> [!IMPORTANT]
> **Reporting Precision**: Never describe PNTC as "free". Always state: *"PNTC introduces zero additional trainable parameters with negligible persistent parameter overhead, incurring a minimal latency delta of $+4.2\text{ ms/sample}$."*
