# PNTC Explain: Validation & Verification Report

This document records the quantitative validation results of the **PNTC Explain** module across calibrated synthetic benchmarks, failure fallback scenarios, and frozen model regression tests.

---

## 1. Quantitative Synthetic Calibration Benchmark

All geometric measurement algorithms were validated against synthetically constructed grids with exact known physical dimensions and depth offsets ($0.4\text{--}0.5\text{ mm/pixel}$ spacing).

| Synthetic Scenario | Ground Truth Parameter | Ground Truth Value | Measured Estimate | Measurement Error | Status ($<5\%$ Tolerance) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Circular Depression** | Diameter | $20.00\text{ mm}$ | $19.38\text{ mm}$ | $3.1\%$ | **PASSED** |
| **Circular Depression** | Maximum Depth | $2.50\text{ mm}$ | $2.50\text{ mm}$ | $0.0\%$ | **PASSED** |
| **Circular Depression** | Projected Area | $314.16\text{ mm}^2$ | $298.54\text{ mm}^2$ | $4.9\%$ | **PASSED** |
| **Rectangular Protrusion** | Major Length (Height)| $16.00\text{ mm}$ | $16.00\text{ mm}$ | $0.0\%$ | **PASSED** |
| **Rectangular Protrusion** | Minor Length (Width) | $10.00\text{ mm}$ | $10.00\text{ mm}$ | $0.0\%$ | **PASSED** |
| **Rectangular Protrusion** | Maximum Protrusion | $3.20\text{ mm}$ | $3.20\text{ mm}$ | $0.0\%$ | **PASSED** |
| **Missing/NaN XYZ Grid** | Physical Calibration | Suppressed | `False` | $0.0\%$ (Graceful Fallback) | **PASSED** |

---

## 2. Unit Test Suite Summary (`tests/explain/`)

The full explanation test suite comprises 20 automated tests executed via `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.6, pytest-9.1.1, pluggy-1.6.0
collected 20 items

tests/explain/test_morphology2d.py::test_circular_disk_morphology PASSED [  5%]
tests/explain/test_morphology2d.py::test_elongated_stripe_morphology PASSED [ 10%]
tests/explain/test_morphology2d.py::test_irregular_shape_solidity PASSED [ 15%]
tests/explain/test_physical_measurements.py::test_calibrated_circular_depression_dimensions PASSED [ 20%]
tests/explain/test_physical_measurements.py::test_calibrated_rectangular_protrusion PASSED [ 25%]
tests/explain/test_physical_measurements.py::test_missing_xyz_graceful_fallback PASSED [ 30%]
tests/explain/test_reasoning.py::test_normal_sample_clean_behavior PASSED [ 35%]
tests/explain/test_reasoning.py::test_dual_modality_evidence_explanation PASSED [ 40%]
tests/explain/test_regions.py::test_empty_anomaly_map PASSED             [ 45%]
tests/explain/test_regions.py::test_single_connected_component PASSED    [ 50%]
tests/explain/test_regions.py::test_multiple_component_ranking PASSED    [ 55%]
tests/explain/test_regions.py::test_noise_spike_filtering PASSED         [ 60%]
tests/explain/test_regression_pntc.py::test_frozen_pntc_canonical_metrics PASSED [ 65%]
tests/explain/test_surface.py::test_localized_depression_surface PASSED  [ 70%]
tests/explain/test_surface.py::test_raised_protrusion_surface PASSED     [ 75%]
tests/explain/test_surface.py::test_appearance_only_flat_surface PASSED  [ 80%]
tests/explain/test_units.py::test_explicit_mm_unit PASSED                [ 85%]
tests/explain/test_units.py::test_explicit_m_unit_scaling PASSED         [ 90%]
tests/explain/test_units.py::test_unknown_auto_detection_meters PASSED   [ 95%]
tests/explain/test_units.py::test_unknown_auto_detection_uncalibrated PASSED [100%]

============================= 20 passed in 2.46s ==============================
```

---

## 3. Regression Invariance Check: Frozen PNTC Metrics

To guarantee that the introduction of PNTC Explain did not modify the frozen detector, metrics were verified directly against canonical test records:

| Metric | Canonical Target | Recomputed After Explain | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Image AUROC ($I\text{-AUROC}$)** | `0.96541000` | `0.96541000` | `0.00000000` | **IDENTICAL / ZERO EFFECT** |
| **Pixel AUROC ($P\text{-AUROC}$)** | `0.99416000` | `0.99416000` | `0.00000000` | **IDENTICAL / ZERO EFFECT** |
| **Region Overlap ($\text{AUPRO@0.3}$)**| `0.96939000` | `0.96939000` | `0.00000000` | **IDENTICAL / ZERO EFFECT** |

---

## 4. Real-Sample Output Verification (`cookie_combined_000`)

```
================================================================================
PNTC DEFECT CHARACTERIZATION REPORT — COOKIE (Sample: cookie_combined_000)
================================================================================
Overall Decision     : ANOMALOUS
Image Anomaly Score  : 0.65715
Detection Confidence : 79.4%
Operating Threshold  : 0.50000
Coordinate Units     : mm (Physical Calibrated: True)
Defect Regions Count : 2
--------------------------------------------------------------------------------

[DEFECT REGION #1] (Importance Rank: #1)
  Location            : central surface (Normalized: (0.6498, 0.3534))
  Bounding Dimensions : 9.69 mm × 8.12 mm (Depth: 0.43 mm)
  Estimated 3D Area   : 68.38 mm² (Projected: 61.85 mm²)
  2D Morphology       : compact circular region (Aspect Ratio: 1.24, Circularity: 1.0)
  3D Surface Structure: rough raised protrusion (Z-score: 8.93, Roughness Ratio: 2.97×)
  Maximum Protrusion  : 0.68 mm
  Appearance Contrast : STRONG (CIE Lab ΔE = 27.52)
  Detection Evidence  : RGB=Moderate (0.56), 3D=Moderate (0.56), Topology Disagreement=High (0.684)
  Measurement Conf.   : 97.0%
  Explanation Rationale:
    "A compact circular region was detected in the central surface, measuring approximately 9.69 mm × 8.12 mm. Visual inspection reveals strong appearance deviation from the normal reference ring (CIE Lab ΔE = 27.52). Both visual and geometric branches contribute consistent anomaly evidence (RGB: Moderate, 3D: Moderate, Topology: High)."
================================================================================
```
