# Paired Neighborhood Topology Consistency (PNTC): Complete Mathematical Formulation

**Document**: `docs/H5_PNTC_FORMULATION.md`  
**Phase**: H5 Formulation Specification  
**Scope**: Exact mathematical definition of Paired Coreset Construction, Retrieval Topology Cues ($T(p)$), Normal Retrieval Confidence Gate ($G(p)$), Hyperparameters, and Decision Fusion.

---

## 1. Paired Feature Space & Normal Coreset Construction

Let the training set of nominal (defect-free) industrial samples be denoted as $\mathcal{D}_{\text{train}}$. For each sample, we extract spatially aligned feature maps for RGB (via frozen DINOv2 ViT-B/14) and 3D (via frozen Point-MAE), mapped to a common physical spatial grid of $H \times W = 56 \times 56 = 3136$ patches.

For each physical patch index $j \in \{1, \dots, N_{\text{train}}\}$ across all training samples:
- $r_j \in \mathbb{R}^{D_{\text{rgb}}}$ is the DINOv2 RGB feature ($D_{\text{rgb}} = 768$).
- $x_j \in \mathbb{R}^{D_{\text{xyz}}}$ is the Point-MAE feature ($D_{\text{xyz}} = 1152$).
- The paired prototype is defined as $P_j = (r_j, x_j)$ with a **single unified prototype identifier** $j$.

### 1.1 Pair-Preserving Joint Coreset Subsampling
To compress the memory bank without breaking cross-modal index alignment:
1. Normalize features:
   $$\hat{r}_j = \frac{r_j}{\|r_j\|_2}, \quad \hat{x}_j = \frac{x_j}{\|x_j\|_2}$$
2. Construct temporary selection vector:
   $$q_j = \left[ \alpha \hat{r}_j \,\|\, \alpha \hat{x}_j \right] \in \mathbb{R}^{D_{\text{rgb}} + D_{\text{xyz}}}, \quad \alpha = \frac{1}{\sqrt{2}}$$
3. Apply greedy MinMax / k-Center coreset selection on $\{q_j\}_{j=1}^{N_{\text{train}}}$ with fixed seed $s \in \{42, 100, 2026\}$ to select an index subset $\mathcal{C} \subset \{1, \dots, N_{\text{train}}\}$ of size $|\mathcal{C}| = M = 15{,}000$.
4. Store separate modality memory banks sharing identical index alignment:
   $$\mathcal{M}_{\text{rgb}} = \{r_j\}_{j \in \mathcal{C}}, \quad \mathcal{M}_{\text{xyz}} = \{x_j\}_{j \in \mathcal{C}}$$

---

## 2. Test Patch Prototype Retrieval

Given an aligned test patch $p$ with features $(r_p, x_p)$:
1. Query $\mathcal{M}_{\text{rgb}}$ for the top-$k$ ($k=5$) nearest prototypes under Euclidean distance:
   $$\mathcal{N}_{\text{rgb}}(p) = \left\{ (j_{\text{rgb}, 1}, d_{\text{rgb}, 1}), \dots, (j_{\text{rgb}, k}, d_{\text{rgb}, k}) \right\}$$
   where $d_{\text{rgb}, m} = \|r_p - r_{j_{\text{rgb}, m}}\|_2$ and $j_{\text{rgb}, m} \in \mathcal{C}$.
2. Query $\mathcal{M}_{\text{xyz}}$ independently for the top-$k$ ($k=5$) nearest prototypes:
   $$\mathcal{N}_{\text{xyz}}(p) = \left\{ (j_{\text{xyz}, 1}, d_{\text{xyz}, 1}), \dots, (j_{\text{xyz}, k}, d_{\text{xyz}, k}) \right\}$$
   where $d_{\text{xyz}, m} = \|x_p - x_{j_{\text{xyz}, m}}\|_2$ and $j_{\text{xyz}, m} \in \mathcal{C}$.

Let $\mathcal{I}_{\text{rgb}}(p) = \{j_{\text{rgb}, 1}, \dots, j_{\text{rgb}, k}\}$ and $\mathcal{I}_{\text{xyz}}(p) = \{j_{\text{xyz}, 1}, \dots, j_{\text{xyz}, k}\}$.

---

## 3. Topology Consistency Cues $T(p)$

### 3.1 Ablation H5-A: Top-$k$ Jaccard Disagreement ($D_{\text{jaccard}}$)
Measures the unweighted set intersection over union of retrieved prototype IDs:
$$\mathcal{J}(p) = \frac{|\mathcal{I}_{\text{rgb}}(p) \cap \mathcal{I}_{\text{xyz}}(p)|}{|\mathcal{I}_{\text{rgb}}(p) \cup \mathcal{I}_{\text{xyz}}(p)|} \in [0, 1]$$
$$T_{\text{jaccard}}(p) = 1 - \mathcal{J}(p)$$

### 3.2 Ablation H5-B: Rank-Weighted Overlap ($D_{\text{rank}}$)
Penalizes mismatches at the top of the ranking more heavily using reciprocal rank weighting:
$$W(p) = \sum_{j \in \mathcal{I}_{\text{rgb}}(p) \cap \mathcal{I}_{\text{xyz}}(p)} \frac{1}{\text{rank}_{\text{rgb}}(j) \cdot \text{rank}_{\text{xyz}}(j)}$$
For $k=5$, the theoretical maximum when $\text{rank}_{\text{rgb}} = \text{rank}_{\text{xyz}} = [1, 2, 3, 4, 5]$ is:
$$W_{\text{max}} = \sum_{m=1}^{5} \frac{1}{m^2} = 1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \frac{1}{25} \approx 1.46361$$
$$T_{\text{rank}}(p) = 1 - \frac{W(p)}{W_{\text{max}}} \in [0, 1]$$

### 3.3 Ablation H5-C: Soft Prototype Jensen-Shannon Divergence ($D_{\text{JS}}$)
Let $\mathcal{U}(p) = \mathcal{I}_{\text{rgb}}(p) \cup \mathcal{I}_{\text{xyz}}(p)$. We construct categorical distributions $P_{\text{rgb}}$ and $P_{\text{xyz}}$ over all $j \in \mathcal{U}(p)$:
$$\tilde{p}_{\text{rgb}}(j) = \begin{cases} \exp\left(-\frac{d_{\text{rgb}}(j)}{\tau_{\text{rgb}}}\right) & \text{if } j \in \mathcal{I}_{\text{rgb}}(p) \\ \epsilon & \text{otherwise} \end{cases}$$
$$\tilde{p}_{\text{xyz}}(j) = \begin{cases} \exp\left(-\frac{d_{\text{xyz}}(j)}{\tau_{\text{xyz}}}\right) & \text{if } j \in \mathcal{I}_{\text{xyz}}(p) \\ \epsilon & \text{otherwise} \end{cases}$$
where $\epsilon = 10^{-6}$, and temperature parameters $\tau_{\text{rgb}}, \tau_{\text{xyz}}$ are set to the median 1-NN distances measured across train-normal samples:
$$\tau_{\text{rgb}} = \text{median}\left( \{d_{\text{rgb}, 1}^{\text{train}}\} \right), \quad \tau_{\text{xyz}} = \text{median}\left( \{d_{\text{xyz}, 1}^{\text{train}}\} \right)$$
Normalize probabilities:
$$p_{\text{rgb}}(j) = \frac{\tilde{p}_{\text{rgb}}(j)}{\sum_{m \in \mathcal{U}(p)} \tilde{p}_{\text{rgb}}(m)}, \quad p_{\text{xyz}}(j) = \frac{\tilde{p}_{\text{xyz}}(j)}{\sum_{m \in \mathcal{U}(p)} \tilde{p}_{\text{xyz}}(m)}$$
Compute Jensen-Shannon Divergence:
$$M(j) = \frac{1}{2}\left(p_{\text{rgb}}(j) + p_{\text{xyz}}(j)\right)$$
$$T_{\text{JS}}(p) = \text{JSD}(p_{\text{rgb}} \,\|\, p_{\text{xyz}}) = \frac{1}{2} D_{\text{KL}}(p_{\text{rgb}} \,\|\, M) + \frac{1}{2} D_{\text{KL}}(p_{\text{xyz}} \,\|\, M) \in [0, \ln 2]$$
Normalized:
$$T_{\text{JS, norm}}(p) = \frac{T_{\text{JS}}(p)}{\ln 2} \in [0, 1]$$

---

## 4. Normal-Derived Retrieval Confidence Gate $G(p)$

The topology discrepancy $T(p)$ is most informative when at least one modality exhibits high-confidence retrieval (i.e. compact normal neighborhood) rather than diffuse background noise.

We define the normal retrieval confidence gate $G(p)$ strictly from train-normal statistics without anomaly labels:
1. Compute the top-1 normalized distance for each modality:
   $$z_{\text{rgb}}(p) = \frac{d_{\text{rgb}, 1}(p) - \mu_{\text{train, rgb}}}{\sigma_{\text{train, rgb}}}, \quad z_{\text{xyz}}(p) = \frac{d_{\text{xyz}, 1}(p) - \mu_{\text{train, xyz}}}{\sigma_{\text{train, xyz}}}$$
2. Measure neighborhood compactness / inverse dispersion:
   $$C(p) = \exp\left( -\frac{\text{std}(\{d_{\text{rgb}, m}\}_{m=1}^k) + \text{std}(\{d_{\text{xyz}, m}\}_{m=1}^k)}{2 \cdot \bar{\sigma}_{\text{dispersion}}} \right) \in (0, 1]$$
3. The confidence gate is formulated as:
   $$G(p) = \text{clip}\left( \sigma\left( \max(z_{\text{rgb}}(p), z_{\text{xyz}}(p)) \right) \cdot C(p), \, 0.0, \, 1.0 \right)$$
   where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the standard sigmoid function.

---

## 5. Final Anomaly Scoring Formulation (H5-D)

Let $A_{\text{base}}(p)$ be the baseline normalized decision fusion score from H4B:
$$A_{\text{base}}(p) = \frac{1}{2} \tilde{A}_{\text{rgb}}(p) + \frac{1}{2} \tilde{A}_{\text{xyz}}(p)$$
where $\tilde{A}$ denotes train-normal z-score standardized and calibrated anomaly maps.

The total PNTC anomaly score at patch $p$ is:
$$A_{\text{pntc}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T_{\text{JS, norm}}(p)$$
with constant hyperparameter $\lambda = 0.35$ (fixed across all 10 categories, selected on train-normal calibration).

### Image-Level Aggregation:
The image anomaly score $S_{\text{image}}$ is computed using the frozen rule from H3.5 / H4:
$$S_{\text{image}} = \text{mean\_top\_0.5\%}(A_{\text{pntc}})$$
smoothing and interpolation to full resolution $(224 \times 224)$ follows the identical Gaussian blur ($\sigma=4$) and bilinear resizing of H4B.
