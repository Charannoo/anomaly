# Paired Neighborhood Topology Consistency for Multimodal Industrial Anomaly Detection

**Document**: `docs/PNTC_METHOD_PAPER.md`  
**Method Name**: Paired Neighborhood Topology Consistency (PNTC)  
**Narrow Novelty Claim**: *Paired normal-prototype retrieval topology consistency for RGB–3D industrial anomaly detection.*

---

## 1. Motivation & Problem Formulation

In unsupervised multimodal industrial inspection (2D RGB images and 3D point clouds), defects are defined by localized departures from nominal manufacturing patterns. Prior memory-bank frameworks (e.g. M3DM, G2SF) compute anomaly evidence via Euclidean or anisotropic distance between test descriptors and the nearest normal prototype in each modality.

However, distance-based scoring encounters a fundamental vulnerability: **manifold aliasing**. 
- An anomalous patch may exhibit surface texture that is individually normal on a different object part (e.g., a perimeter rim shadow appearing in the center).
- Simultaneously, its 3D depth may resemble nominal curvature from yet another distinct part.

Under independent distance scoring, both modalities observe small nearest-neighbor distances, producing a false negative.

**PNTC's Core Thesis**:
Nominal patches describing the *same physical surface location* must map to *compatible local neighborhoods* on the shared normal manifold. Even if unimodal distances are small, an anomaly disrupts the joint manifold topology, causing RGB and 3D queries to retrieve divergent, incompatible normal prototype clusters.

---

## 2. Paired Prototype Normal Memory Bank

Let $\mathcal{D}_{\text{train}} = \{(\mathbf{I}_i, \mathbf{P}_i)\}_{i=1}^{N_{\text{samples}}}$ denote the training set of nominal industrial items. Using frozen feature backbones (DINOv2 ViT-B/14 for RGB, Point-MAE for 3D), we extract spatially aligned feature maps on a common spatial grid of $H \times W = 56 \times 56 = 3136$ patches.

For each physical patch index $j \in \{1, \dots, N_{\text{train}}\}$ across all training samples:
- $r_j \in \mathbb{R}^{D_{\text{rgb}}}$ is the DINOv2 visual descriptor ($D_{\text{rgb}} = 768$).
- $x_j \in \mathbb{R}^{D_{\text{xyz}}}$ is the Point-MAE geometric descriptor ($D_{\text{xyz}} = 1152$).
- The paired prototype is defined as $P_j = (r_j, x_j)$, sharing a **single unified prototype identifier** $j$.

### Joint Pair-Preserving Coreset Subsampling
To compress the memory bank while strictly preserving cross-modal indexing:
1. Normalize descriptors: $\hat{r}_j = r_j / \|r_j\|_2$, $\hat{x}_j = x_j / \|x_j\|_2$.
2. Form a temporary joint selection vector:
   $$q_j = \left[ \frac{1}{\sqrt{2}} \hat{r}_j \,\|\, \frac{1}{\sqrt{2}} \hat{x}_j \right] \in \mathbb{R}^{D_{\text{rgb}} + D_{\text{xyz}}}$$
3. Apply greedy MinMax coreset selection on $\{q_j\}$ to select an index subset $\mathcal{C} \subset \{1, \dots, N_{\text{train}}\}$ of size $|\mathcal{C}| = M = 15{,}000$.
4. Form paired memory banks:
   $$\mathcal{M}_{\text{rgb}} = \{r_j\}_{j \in \mathcal{C}}, \quad \mathcal{M}_{\text{xyz}} = \{x_j\}_{j \in \mathcal{C}}$$

Crucially, $\mathcal{M}_{\text{rgb}}[m]$ and $\mathcal{M}_{\text{xyz}}[m]$ represent the *exact same physical patch* on the normal manifold for all $m \in \{1, \dots, M\}$.

---

## 3. Independent Prototype Retrieval & Neighborhood Topology

Given a test sample with aligned feature pair $(r_p, x_p)$ at patch $p$:
1. Query $\mathcal{M}_{\text{rgb}}$ for the top-$k$ ($k=5$) nearest prototypes under Euclidean metric:
   $$\mathcal{N}_{\text{rgb}}(p) = \left\{ (j_{\text{rgb}, 1}, d_{\text{rgb}, 1}), \dots, (j_{\text{rgb}, k}, d_{\text{rgb}, k}) \right\}$$
2. Query $\mathcal{M}_{\text{xyz}}$ independently for the top-$k$ ($k=5$) nearest prototypes:
   $$\mathcal{N}_{\text{xyz}}(p) = \left\{ (j_{\text{xyz}, 1}, d_{\text{xyz}, 1}), \dots, (j_{\text{xyz}, k}, d_{\text{xyz}, k}) \right\}$$

Let $\mathcal{I}_{\text{rgb}}(p) = \{j_{\text{rgb}, 1}, \dots, j_{\text{rgb}, k}\}$ and $\mathcal{I}_{\text{xyz}}(p) = \{j_{\text{xyz}, 1}, \dots, j_{\text{xyz}, k}\}$.

### Soft Prototype Distribution Divergence ($T(p)$)
Let $\mathcal{U}(p) = \mathcal{I}_{\text{rgb}}(p) \cup \mathcal{I}_{\text{xyz}}(p)$ be the union of retrieved prototype IDs. We construct categorical probability distributions $P_{\text{rgb}}$ and $P_{\text{xyz}}$ over all $j \in \mathcal{U}(p)$:
$$\tilde{p}_{\text{rgb}}(j) = \begin{cases} \exp\left( - \frac{d_{\text{rgb}}(j)}{\tau_{\text{rgb}}} \right) & \text{if } j \in \mathcal{I}_{\text{rgb}}(p) \\ \epsilon & \text{otherwise} \end{cases}$$
$$\tilde{p}_{\text{xyz}}(j) = \begin{cases} \exp\left( - \frac{d_{\text{xyz}}(j)}{\tau_{\text{xyz}}} \right) & \text{if } j \in \mathcal{I}_{\text{xyz}}(p) \\ \epsilon & \text{otherwise} \end{cases}$$
where $\epsilon = 10^{-6}$, and temperatures $\tau_{\text{rgb}}, \tau_{\text{xyz}}$ are the median 1-NN distances measured across nominal training features.

Normalizing over $\mathcal{U}(p)$ yields $P_{\text{rgb}}(j) = \tilde{p}_{\text{rgb}}(j) / \sum_{m} \tilde{p}_{\text{rgb}}(m)$ and $P_{\text{xyz}}(j) = \tilde{p}_{\text{xyz}}(j) / \sum_{m} \tilde{p}_{\text{xyz}}(m)$.

The continuous topology inconsistency cue is computed as the normalized Jensen-Shannon Divergence:
$$M(j) = \frac{1}{2}\left(P_{\text{rgb}}(j) + P_{\text{xyz}}(j)\right)$$
$$T(p) = \frac{\text{JSD}(P_{\text{rgb}} \,\|\, P_{\text{xyz}})}{\ln 2} = \frac{D_{\text{KL}}(P_{\text{rgb}} \,\|\, M) + D_{\text{KL}}(P_{\text{xyz}} \,\|\, M)}{2 \ln 2} \in [0, 1]$$

---

## 4. Normal-Derived Retrieval Confidence Gate $G(p)$

Disagreement between modalities is informative only when retrieval reflects localized manifold structure rather than diffuse background noise. We derive a retrieval confidence gate $G(p)$ strictly from nominal training statistics:
1. Standardize 1-NN distances against nominal training means and standard deviations:
   $$z_{\text{rgb}}(p) = \frac{d_{\text{rgb}, 1}(p) - \mu_{\text{train, rgb}}}{\sigma_{\text{train, rgb}}}, \quad z_{\text{xyz}}(p) = \frac{d_{\text{xyz}, 1}(p) - \mu_{\text{train, xyz}}}{\sigma_{\text{train, xyz}}}$$
2. Measure neighborhood compactness via top-$k$ dispersion:
   $$C(p) = \exp\left( -\frac{\text{std}(\{d_{\text{rgb}, m}\}_{m=1}^k) + \text{std}(\{d_{\text{xyz}, m}\}_{m=1}^k)}{2 \cdot \bar{\sigma}_{\text{dispersion}}} \right) \in (0, 1]$$
3. Gate formula:
   $$G(p) = \text{clip}\left( \sigma\left(\max(z_{\text{rgb}}(p), z_{\text{xyz}}(p))\right) \cdot C(p), \, 0.0, \, 1.0 \right)$$

---

## 5. Decision Fusion & Evaluation

The overall anomaly score map $A_{\text{pntc}}(p)$ combines base distance evidence with gated topology inconsistency:
$$A_{\text{pntc}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$$
where $A_{\text{base}}(p)$ is the calibrated z-score decision fusion score from H4B, and $\lambda = 0.35$ is a constant scaling hyperparameter.

The pixel map is smoothed via Gaussian filter ($\sigma = 3.8$) and upsampled to $224 \times 224$. The image anomaly score is computed as:
$$S_{\text{image}} = \text{mean\_top\_0.5\%}(A_{\text{pntc}})$$
