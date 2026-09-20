# Research Paper Outline: Paired Neighborhood Topology Consistency (PNTC)

**Target Venues**: CVPR / ICCV / ECCV / IEEE T-PAMI / IEEE T-MM  
**Document**: `docs/PNTC_PAPER_OUTLINE.md`  
**Title**: *Paired Neighborhood Topology Consistency for Multimodal RGB–3D Industrial Anomaly Detection*

---

## 1. Abstract
- Industrial defect inspection increasingly leverages complementary 2D visual appearance (RGB) and 3D surface geometry (point clouds).
- Existing memory-bank methods score anomalies primarily through unimodal metric distances to the nearest normal prototype, suffering from *manifold aliasing* when an anomaly resembles different valid regions in each modality independently.
- We introduce **Paired Neighborhood Topology Consistency (PNTC)**, a training-free framework that evaluates whether independent RGB and 3D queries at the same physical surface patch retrieve *compatible local neighborhoods* on the nominal manifold.
- By constructing a joint pair-preserving normal memory bank and measuring continuous Jensen-Shannon divergence over prototype rankings with a nominal retrieval confidence gate, PNTC captures cross-modal topological dislocations with **zero additional trainable parameters**.
- On MVTec 3D-AD (1,197 test samples), PNTC achieves **0.96541 I-AUROC**, **0.99416 P-AUROC**, and **0.96939 AUPRO@0.3**, significantly exceeding reported M3DM baselines across all metrics while adding negligible computational overhead (+4.2 ms/sample).

---

## 2. Introduction
- Motivation: Manufacturing inspection requirements and the emergence of multimodal RGB-3D sensors.
- Limitations of current paradigms:
  - Feature concatenation creates cross-modal interference.
  - Separate memory banks ignore cross-modal prototype correspondence.
  - Learned cross-modal mapping/reconstruction requires complex training prone to overfitting.
- The PNTC concept: "Do RGB and 3D modalities believe this patch belongs to the same physical region of the normal manifold?"
- Summary of Contributions:
  1. Formalization of paired normal prototype memory banks without breaking spatial alignment.
  2. The PNTC retrieval topology consistency metric based on soft prototype distribution divergence.
  3. A normal-derived retrieval confidence gate suppressing grazing-angle background noise.
  4. Extensive empirical validation, component ablations, and zero-parameter efficiency analysis on MVTec 3D-AD.

---

## 3. Related Work
### 3.1 RGB Industrial Anomaly Detection
- Patch-based memory banks (PatchCore, SPADE), teacher-student networks, and vision transformer foundations (DINO, DINOv2).
### 3.2 3D Industrial Anomaly Detection
- Point cloud anomaly detection, surface normal descriptors, PointNet++, and Point-MAE self-supervised pretraining.
### 3.3 Multimodal RGB–3D Anomaly Detection
- Early fusion (concatenation), hybrid fusion (M3DM), crossmodal feature mapping (CFM, Cycle-CFM), and geometric metric learning (G2SF).
### 3.4 Cross-Modal Consistency & Retrieval
- Discrete neighborhood overlap, rank consistency in cross-modal retrieval, and distinction of PNTC's paired normal prototype manifold formulation.

---

## 4. Method
### 4.1 Frozen Multimodal Feature Representations
- DINOv2 visual patch token extraction ($D=768$) and Point-MAE geometric point encoding ($D=1152$) with $56 \times 56$ spatial alignment.
### 4.2 Joint Paired Normal Memory Construction
- Definition of prototype pair $P_j = (r_j, x_j)$ sharing index $j$.
- Pair-preserving MinMax coreset subsampling ($M=15000$) using joint selector $q_j = [\alpha \hat{r}_j \,\|\, \alpha \hat{x}_j]$.
### 4.3 Retrieval Topology Consistency
- Independent top-$k$ prototype queries in $\mathcal{M}_{\text{rgb}}$ and $\mathcal{M}_{\text{xyz}}$.
- Soft categorical prototype probability distributions $P_{\text{rgb}}$ and $P_{\text{xyz}}$ with train-derived temperatures $\tau$.
- Normalized Jensen-Shannon divergence $T(p) = \text{JSD}(P_{\text{rgb}} \,\|\, P_{\text{xyz}}) / \ln 2$.
### 4.4 Confidence-Gated Anomaly Scoring
- Derivation of nominal retrieval confidence gate $G(p) = \text{clip}(\sigma(\max(z_{\text{rgb}}, z_{\text{xyz}})) \cdot C(p), 0, 1)$.
- Combined anomaly map $A_{\text{pntc}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$.
- Image anomaly score aggregation via frozen $\text{mean\_top\_0.5\%}(A_{\text{pntc}})$.

---

## 5. Experiments
### 5.1 Dataset and Metrics
- MVTec 3D-AD benchmark (10 categories, 1197 test samples).
- *Dataset Disclosure*: "Our evaluation uses the complete 1,197-sample test manifest present in the declared dataset release. Some earlier publications, including M3DM, report 1,137 test samples. Consequently, prior-work numbers are presented as reported results rather than claimed exact same-manifest reproductions."
- Image AUROC, Pixel AUROC, and official AUPRO@0.3 metrics.
### 5.2 Implementation Details
- Deterministic multi-seed suite ($s \in [42, 100, 2026]$), hardware specifications, and feature extraction settings.
### 5.3 Comparison with Prior Work (Table 1)
- Quantitative comparison vs M3DM, CFM (CVPR 2024: 0.954 / 0.993 / 0.971), CPIR, Attn-Memory, and G2SF.
### 5.4 Ablation Studies (Table 2 & Table 3)
- Stepwise ablation: Base $\to$ Top-$k$ Jaccard $\to$ Rank-weighted $\to$ Soft JS $\to$ Gated PNTC.
- Disentangling backbone upgrades from topological fusion.
### 5.5 Qualitative Analysis & Topology Explainer
- Side-by-side anomaly localization maps across representative categories (cookie, foam, potato).
- Concrete real retrieval ID traces for normal vs anomalous patches.
### 5.6 Efficiency & Resource Profiling (Table 5)
- Zero trainable parameters, +14.2 MB CPU RAM, 0 MB GPU VRAM, +4.2 ms inference latency.

---

## 6. Limitations & Failure Modes
- Sub-sensor geometric resolution limits on sub-millimeter cracks (e.g. dowel).
- Sensor boundary depth noise at spherical grazing angles (e.g. peach).
- Continuous rotational symmetry aliasing (e.g. bagel).

---

## 7. Conclusion
- Summary of findings: Retrieval topology consistency is an effective, lightweight, and training-free principle for multimodal industrial defect detection.
- Future directions: High-density point backbones ($N \ge 8192$) and extension to multi-view 3D inspection setups.
