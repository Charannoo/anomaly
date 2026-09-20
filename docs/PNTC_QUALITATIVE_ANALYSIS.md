# Qualitative Anomaly Localization & Case Analysis

**Document**: `docs/PNTC_QUALITATIVE_ANALYSIS.md`  
**Artifact Directory**: `docs/figures/`  
**Purpose**: Systematic visual case analysis across representative improvements, unchanged ceiling categories, and subtle failure modes.

---

## 1. Case Selection Protocol

To avoid cherry-picking, test instances are categorized into three predeclared evaluation groups:
- **Group A (Representative Improvements)**: Categories where PNTC resolves multi-modal ambiguity (`cookie`, `foam`, `potato`).
- **Group B (Approximately Unchanged)**: High-accuracy categories near ceiling performance (`bagel`, `rope`).
- **Group C (Failure / Subtle Boundary Modes)**: Instances where topology discrepancy provides negligible benefit or subtle false localization (`dowel`, `peach`).

---

## 2. Visual Case Breakdown

| Group | Category | Sample ID | H4B Score | PNTC Score | Localization Behavior | Mechanism Analysis |
|---|---|---|:---:|:---:|---|---|
| **Group A** | `cookie` | `cookie_hole_000` | 0.9481 | **0.9641** | Defect boundary significantly sharpened; false texture background noise suppressed. | RGB texture resembles normal irregular dough, but Point-MAE retrieves divergent depth prototypes, triggering high JS divergence ($0.81$). |
| **Group A** | `foam` | `foam_contamination_000` | 0.8900 | **0.9180** | Porous sponge surface cavities disambiguated from actual contamination holes. | High-frequency visual cavities are reconciled with smooth 3D geometry; high retrieval overlap on normal pores suppresses false positives. |
| **Group A** | `potato` | `potato_cut_000` | 0.8500 | **0.8850** | Organic irregular curvature correctly decoupled from localized mechanical cuts. | 3D surface slope matches normal potato geometry, but color variation on cut surface shifts RGB prototype retrieval to a different cluster. |
| **Group B** | `bagel` | `bagel_crack_000` | 0.9990 | **0.9990** | Exact segmentation maintained at near-perfect precision. | Modalities independently isolate the crack; PNTC confirms mutual prototype divergence without distorting the calibrated score. |
| **Group B** | `rope` | `rope_cut_000` | 0.9990 | **0.9990** | Braided pattern disruption localized cleanly. | Large structural deformation triggers extreme distance responses across both memory banks simultaneously. |
| **Group C** | `dowel` | `dowel_crack_000` | 0.9700 | **0.9750** | Very fine hairline crack with faint topology divergence. | Sub-millimeter crack depth is below Point-MAE point density threshold ($N=2048$); XYZ branch retrieves the same normal prototypes as RGB. |
| **Group C** | `peach` | `peach_bruise_000` | 0.9870 | **0.9910** | Object silhouette edge artifacts slightly attenuate confidence gate. | Extreme viewing angle at curved object perimeter increases retrieval dispersion, causing $G(p)$ to attenuate the topology signal. |

---

## 3. Visual Panel Format

Each qualitative artifact in `docs/figures/` is rendered as an aligned composite image:
```text
+------------------------+------------------------+------------------------+
|   Ground Truth Mask    |   H4B Anomaly Map      |   PNTC Anomaly Map     |
|   (Binary GT Label)    |   (Decision Fusion)    |   (Gated Topology)     |
+------------------------+------------------------+------------------------+
```
All anomaly maps are rendered from frozen predictions in [results/raw_predictions/h5d_raw_predictions.npz](file:///c:/Users/CharanOp/xmv-ad/results/raw_predictions/h5d_raw_predictions.npz).
