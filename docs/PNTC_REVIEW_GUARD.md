# PNTC Review Guard: Uncertainty Quantification & Manual Review Mode

The **PNTC Review Guard** is an operational deployment safeguard that identifies when inspection results are unreliable, alerting human operators to review borderline or sensor-compromised cases.

---

## 1. Separation of Uncertainties

Industrial decision-making requires distinguishing between detection certainty, physical measurement reliability, and sensor quality.

| Uncertainty Tier | What It Measures | Scale / Metrics | Not Dependent On |
|---|---|---|---|
| **Tier 1: Detection Certainty** | Margin of PNTC anomaly score from operating threshold | `High`, `Moderate`, `Borderline` ($|A - \tau| / \sigma_{\text{norm}}$) | 3D surface mesh residual |
| **Tier 2: Measurement Reliability** | Confidence in physical depth, area, and volume estimates | `High`, `Moderate`, `Low` (factors XYZ valid ratio & plane fit residual) | Anomaly score magnitude |
| **Tier 3: Sensor Data Quality** | Health of raw input signals | Valid point fraction, RGB contrast, silhouette edge proximity | Anomaly label |

> [!WARNING]
> **NO UNCALIBRATED PROBABILITIES**:
> The system does not manufacture arbitrary probabilities (e.g. "94% confident"). It outputs mathematically grounded margins and diagnostic metrics.

---

## 2. Non-Modification of Canonical PNTC Decisions

The Review Guard is strictly a deployment triage layer:
- `pntc_decision`: Stays permanently equal to the canonical binary decision (`"normal"` vs `"anomalous"`).
- `inspection_status`: Operational routing recommendation:
  - `ACCEPT_NORMAL`: Normal sample conforming to nominal memory with high certainty.
  - `DEFECT_DETECTED`: Clear defect with robust sensor coverage and high decision certainty.
  - `MANUAL_REVIEW_RECOMMENDED`: Detection or geometric estimation flagged by one or more diagnostic warnings.

Benchmark AUROC and AUPRO evaluations evaluate `pntc_decision` and raw anomaly maps exclusively, guaranteeing zero benchmark leakage.

---

## 3. Deterministic Review Triggers

The system monitors 9 deterministic diagnostic conditions:

1. **`LOW_XYZ_COVERAGE`**: Valid 3D points inside defect region $< 60\%$ (critical trigger if $< 50\%$).
2. **`LOW_RGB_QUALITY`**: Standard deviation of pixel values $< 0.08 \times \mu$, indicating severe under-exposure or saturation.
3. **`NEAR_DECISION_THRESHOLD`**: Anomaly score lies within $0.8 \sigma_{\text{train}}$ of operating threshold.
4. **`UNSTABLE_NEIGHBOR_RETRIEVAL`**: Top-k retrieval margin gap $d_{k+1} - d_k < 0.015$, indicating prototype ambiguity.
5. **`LOW_SURFACE_FIT_CONFIDENCE`**: Reference surface fit residual is elevated (confidence $< 0.60$).
6. **`SMALL_REGION`**: Defect area $< 15\,\text{px}$, where single-pixel sensor noise could skew morphology.
7. **`OBJECT_BOUNDARY_REGION`**: Defect centroid lies adjacent to object silhouette edge ($> 0.75$), where organized point cloud depth drops off.
8. **`CROSS_MODAL_ALIGNMENT_WARNING`**: Misalignment between RGB texture and 3D depth.
9. **`MISSING_PHYSICAL_CALIBRATION`**: XYZ coordinates lack verified physical millimeter scaling.

---

## 4. Manual Review Output Example

```
MANUAL INSPECTION RECOMMENDED

Reason:
PNTC detected an abnormal region (score: 0.518 vs threshold: 0.500), but the defect 
lies close to the object's boundary and only 58% of its XYZ samples are valid.

The canonical anomaly detection decision (ANOMALOUS) is strictly preserved, but 
geometric size and volume estimates should be verified manually.
```
