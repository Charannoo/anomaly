# PNTC Normal Twin: Retrieval-Based Normal Analogue Reference

The **PNTC Normal Twin** is a post-detection explanation component that identifies and visualizes the closest paired normal references from the frozen training memory for an anomalous region.

---

## 1. Scientific Concept: Retrieved Normal Analogue vs. Generative Reconstruction

> [!IMPORTANT]
> **CRITICAL SCIENTIFIC DISTINCTION**:
> The Normal Twin is **NOT** a generated reconstruction. The system makes no claim of synthesizing the hypothetical "defect-free version" of the test sample via a diffusion model, GAN, or autoencoder.
> 
> Instead, it is strictly a:
> **"retrieved normal analogue"** or **"nearest paired normal reference"**.

### Why Retrieval Rather Than Generative Synthesis?
1. **Verifiable Provenance**: Every retrieved normal reference exists physically in the curated nominal training dataset. Its source sample ID, object category, and spatial patch coordinates are 100% auditable.
2. **Zero Hallucination Risk**: Generative models frequently hallucinate non-existent features or blend contradictory structures when reconstructing missing/defective industrial parts. Retrieval guarantees that the comparison reference is a genuine, physically feasible nominal sample.
3. **No Inference-Time Retraining**: The retrieval uses the exact frozen paired prototype bank constructed during PNTC training ($M = 15{,}000$ coreset prototypes).

---

## 2. Provenance Data Model

Every prototype $j \in \{0, \dots, M-1\}$ in the memory bank is indexed with complete historical provenance:

```json
{
  "prototype_id": 1842,
  "category": "cookie",
  "training_sample_id": "train/cookie/good/023",
  "spatial_coord": [24, 38],
  "rgb_bbox": [96, 152, 112, 168],
  "xyz_bbox": [96, 152, 112, 168],
  "rgb_xyz_aligned": true,
  "is_train_normal": true
}
```

### Provenance Invariant Guarantees:
- **Purity**: Zero test samples or anomalous images exist in the normal memory bank.
- **Physical Pairing**: Prototype $j$ corresponds to the identical physical sub-region across both RGB and 3D modalities, preserving cross-modal manifold integrity.

---

## 3. Defect-to-Prototype Mapping Strategy

For any detected connected defect component, the system evaluates representative patches rather than arbitrarily selecting favorable pixels:
1. **Peak Anomaly Evidence Patch**: The patch exhibiting maximum continuous anomaly score within the defect mask ($p_{\text{evidence}} = \arg\max_{p \in \mathcal{D}} A_{\text{PNTC}}(p)$).
2. **Peak Topology Disagreement Patch**: The patch exhibiting maximum cross-modal Jensen-Shannon divergence ($p_{\text{topo}} = \arg\max_{p \in \mathcal{D}} T_{\text{JS}}(p)$).
3. **Region Medoid Patch**: The geometric center of the defect mask closest to the centroid, representing typical defect morphology.

For each representative patch:
- Query RGB memory bank for top-$k$ nearest prototypes: $\mathcal{N}_{\text{rgb}}(p) = \{(j_{\text{rgb}, m}, d_{\text{rgb}, m})\}_{m=1}^k$.
- Query 3D memory bank for top-$k$ nearest prototypes: $\mathcal{N}_{\text{xyz}}(p) = \{(j_{\text{xyz}, m}, d_{\text{xyz}, m})\}_{m=1}^k$.
- Evaluate joint reference compatibility score:
  $$D_{\text{joint}}(j) = 0.5 \cdot d_{\text{rgb}}(j) + 0.5 \cdot d_{\text{xyz}}(j)$$
  The prototype minimizing $D_{\text{joint}}(j)$ across the candidate union is designated as the primary Normal Twin.

---

## 4. Visual Comparison & Difference Map

The Normal Twin generates a side-by-side comparative inspection panel:
1. **Observed RGB Crop** vs. **Normal Reference RGB Crop** vs. **Measurable Color Difference ($\Delta E_{Lab}$ / Absolute RGB Error)**.
2. **Observed Surface Depth (Z)** vs. **Expected Local Reference Surface** vs. **Signed Surface Difference Field ($\delta_i$)**.

### Alignment Confidence Metric
Because industrial parts may exhibit minor pose or spatial translation differences:
$$\text{Alignment Confidence} = \text{valid\_xyz\_fraction} \cdot \left(1.0 - 0.4 \cdot \text{boundary\_proximity}\right)$$
When alignment confidence is low ($< 0.60$), visual difference metrics are interpreted cautiously.

---

## 5. Limitations

- **Pose Discrepancies**: If a test sample has a significant rigid-body rotation relative to training samples, the retrieved prototype may exhibit a differing local surface orientation.
- **Normal Intra-Class Variation**: For organic or variable industrial categories (e.g. `carrot` or `potato`), nominal texture variations may yield moderate RGB distances even on defect-free surfaces.
