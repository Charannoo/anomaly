# Demo Cases Selection Rules & Case Manifest

This document records the selection criteria and rationale for the 6 demonstration bundles generated under `results/demo_cases/`.

## Selection Discipline
To prevent cherry-picking and evaluate the inspection suite under realistic industrial conditions, samples were chosen according to explicit diagnostic archetypes:

1. **Case 1 (Strong Defect)**: Clear multimodal anomaly where both RGB contamination and 3D surface depression are strongly pronounced. Expected to exhibit High certainty and High measurement reliability.
2. **Case 2 (Primarily RGB Anomaly)**: Color stain/discoloration where geometry is nominal. Demonstrates that 3D volume integration correctly suppresses false protrusion/depression claims (<0.5 mm³).
3. **Case 3 (Primarily Geometric Anomaly)**: Structural deformation (dent/protrusion) with uniform nominal texture. Demonstrates that 3D volume estimation captures physical material loss/gain even when appearance evidence is modest.
4. **Case 4 (Strong Topology Disagreement)**: Region where individual modalities retrieve normal prototypes, but their cross-modal manifold neighborhoods diverge strongly (JS divergence > 0.90), demonstrating the core research intuition behind PNTC.
5. **Case 5 (Borderline / Manual Review)**: Sample with anomaly score lying immediately adjacent to the operating decision threshold (0.518 vs 0.500) and partial point coverage (58%). Demonstrates automatic escalation to `MANUAL_REVIEW_RECOMMENDED`.
6. **Case 6 (Failure / Limitation Example)**: Defect located along the object silhouette edge with only 42% valid XYZ coverage and boundary proximity > 0.85. Illustrates how the Review Guard prevents misleading physical volume claims by flagging `OBJECT_BOUNDARY_REGION` and `LOW_XYZ_COVERAGE`.

---

## Case Manifest Summary

| Case Directory | Title | Category | PNTC Score | Inspection Status | Certainty | Reliability | Key Warnings |
|---|---|---|---|---|---|---|---|
| `01_strong_defect/` | Strong Multimodal Defect | `cookie` | 0.880 | `DEFECT_DETECTED` | High | High | None |
| `02_primarily_rgb_defect/` | Primarily RGB Color/Texture Anomaly | `peach` | 0.760 | `DEFECT_DETECTED` | High | High | None |
| `03_primarily_geometric_defect/` | Primarily Geometric Defect | `foam` | 0.840 | `DEFECT_DETECTED` | High | High | None |
| `04_strong_topology_disagreement/` | Strong Topology Disagreement | `cable_gland` | 0.740 | `DEFECT_DETECTED` | Moderate | High | None |
| `05_borderline_manual_review/` | Borderline Review Triggered | `bagel` | 0.518 | `MANUAL_REVIEW_RECOMMENDED` | Borderline | Moderate | `NEAR_DECISION_THRESHOLD`, `LOW_XYZ_COVERAGE` |
| `06_failure_limitation_boundary/` | Boundary Artefact Limitation | `dowel` | 0.610 | `MANUAL_REVIEW_RECOMMENDED` | Moderate | Low | `LOW_XYZ_COVERAGE`, `OBJECT_BOUNDARY_REGION` |
