# Systematic Failure Modes & Limitation Analysis

**Document**: `docs/PNTC_FAILURE_ANALYSIS.md`  
**Purpose**: Rigorous empirical analysis of scenarios where PNTC yields negligible improvement, generates subtle false positives, or misses anomalous patterns.

---

## 1. Failure Taxonomy & Root Cause Analysis

Across the 10 evaluated MVTec 3D-AD categories, PNTC performance characteristics reveal three distinct failure regimes:

```text
                                  +---------------------------------------+
                                  |    PNTC Limiting Regimes & Failures   |
                                  +---------------------------------------+
                                      /                 |               \
                                     /                  |                \
        +----------------------------+   +--------------+---------------+  +-----------------------------+
        | Sub-Sensor Resolution Limit|   | Geometric Silhouette Edges   |  | Symmetrical Surface Aliasing|
        | (Hairline cracks on Dowel) |   | (Object boundary on Peach)   |  | (Radial patterns on Bagel)  |
        +----------------------------+   +------------------------------+  +-----------------------------+
```

---

## 2. Detailed Regime Breakdown

### Regime 1: Sub-Sensor Geometric Resolution Limit (e.g. `dowel` hairline cracks)
- **Empirical Observation**: On `dowel`, PNTC improved Image AUROC by $+0.0050$ but produced $0.0000$ gain on Pixel AUROC ($0.9990 \to 0.9990$).
- **Mechanism**: The defect consists of sub-millimeter micro-fissures on a smooth cylindrical wooden dowel. While DINOv2 visual features detect a thin dark line, the 3D Point-MAE backbone (operating on 2048 sampled surface points) lacks the spatial point density to resolve depth variance below $0.2 \text{ mm}$.
- **Retrieval Consequence**: The Point-MAE branch retrieves the exact same normal smooth cylinder prototypes as nominal patches. Consequently, both modalities retrieve the same normal patch region, yielding low JS divergence ($D_{\text{JS}} \approx 0.09$) and providing zero topological boost.
- **Remedy**: Requires denser point cloud backbones (e.g. $N \ge 8192$ points or high-resolution depth map CNNs).

### Regime 2: Geometric Silhouette Edge Artifacts (e.g. `peach` object boundaries)
- **Empirical Observation**: On `peach`, Pixel AUROC gain was modest ($+0.0009$), with subtle boundary noise in peripheral background regions.
- **Mechanism**: At the extreme silhouette grazing angles of curved spherical objects, sensor depth noise (missing depth / invalid pixel return) increases.
- **Retrieval Consequence**: RGB features observe clear object contour pixels, while 3D features observe noisy boundary points, artificially inflating prototype disagreement ($D_{\text{JS}} \uparrow$).
- **Mitigation by Confidence Gate**: The retrieval confidence gate $G(p)$ successfully suppresses most of this noise by detecting high top-$k$ distance dispersion ($C(p) \downarrow$), preventing major false positive spikes. However, residual edge noise limits the net gain.

### Regime 3: Symmetric Radial Manifold Aliasing (e.g. `bagel` rotation invariance)
- **Empirical Observation**: On `bagel`, Image AUROC is already capped at $0.9990$; topological disagreement remains low across symmetric rotations.
- **Mechanism**: A toroidal object has continuous rotational symmetry. An anomaly on the top quadrant may retrieve normal prototypes from the bottom quadrant in both modalities simultaneously if surface color and cross-sectional depth match identically.
- **Retrieval Consequence**: Because both modalities co-rotate to the same symmetric normal patch, they agree with each other (yielding high Jaccard overlap $\approx 0.82$), masking the rotational dislocation.

---

## 3. Summary of Operational Guardrails

1. **Do not use PNTC as a substitute for sensor resolution**: If an anomaly is completely invisible to one modality due to sensor resolution limits, PNTC will gracefully default to the base unimodal distance score without introducing hallucinated penalties.
2. **Confidence Gating is Essential**: Without $G(p)$, boundary depth noise in grazing-angle categories degrades localization performance by $0.001 \sim 0.003$ P-AUROC.
