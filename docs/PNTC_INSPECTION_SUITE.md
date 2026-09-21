# PNTC Industrial Inspection Suite: Architecture & System Specification

The **PNTC Industrial Inspection Suite** is a post-detection operational layer extending the frozen **Paired Neighborhood Topology Consistency (PNTC)** detector. It operates entirely downstream of the frozen anomaly scoring and morphological explanation engines, providing interpretable, physical, and verifiable diagnostic intelligence for industrial quality control.

---

## 1. System Architecture & End-to-End Pipeline

```
             RGB IMAGE                              XYZ POINT CLOUD
                 │                                         │
              DINOv2                                   Point-MAE
                 │                                         │
          RGB representation                       XYZ representation
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                                  PNTC CORE
                           (Frozen Detector: H5-D)
                                      │
                                      ▼
                             ANOMALY SCORE & MAP
                                      │
                                      ▼
                                 PNTC EXPLAIN
                       (Morphology & Region Extraction)
                                      │
                                      ▼
                           DEFECT CHARACTERIZATION
                                      │
         ┌───────────────────┬────────┴──────────┬───────────────────┐
         │                   │                   │                   │
         ▼                   ▼                   ▼                   ▼
    FEATURE 1           FEATURE 2           FEATURE 3           FEATURE 4
   NORMAL TWIN          GEOMETRY            PROTOTYPE            REVIEW
    REFERENCE         QUANTIFICATION          TRACE               GUARD
(Retrieved Normal   (Triangulated 3D      (Top-k Retrieval     (Uncertainty &
    Analogue)         Volume Loss/Gain)   & Score Decomp.)     Manual Triage)
         │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┘
                             │
                             ▼
              UNIFIED INDUSTRIAL INSPECTION REPORT
             (Structured JSON + Human-Readable ASCII)
```

---

## 2. Core Invariance Principles

1. **Frozen Canonical Detector**:
   - The detector weights, features, backbones (DINOv2 ViT-B/14, Point-MAE), hyperparameters ($k=5, \lambda=0.35, M=15000$), decision fusion, and scoring remain strictly untouched.
   - Canonical verified benchmark performance remains bit-exact:
     $$\text{I-AUROC} = 0.96541000, \quad \text{P-AUROC} = 0.99416000, \quad \text{AUPRO@0.3} = 0.96939000$$
2. **Zero Test-Anomaly Tuning**:
   - All thresholds, calibration parameters, and diagnostic rules are derived solely from training nominal samples and first-principles geometry.
3. **Strict Scientific Language Discipline**:
   - **Normal Twin**: Designated strictly as a *retrieved normal analogue* or *nearest paired normal reference*; never described as a generative reconstruction.
   - **Geometry**: Evaluated as *localized depression* or *raised protrusion*; never mapped to unvalidated semantic classes such as "dent" or "crack".
   - **Units**: Physical volume ($\text{mm}^3$) is reported only when coordinates are verified to be physically calibrated; uncalibrated inputs output native units.
   - **Certainty**: Described as *decision certainty* and *measurement reliability*; never converted to uncalibrated percentage probabilities.

---

## 3. Four Core Inspection Features

### Feature 1: Normal Twin Reference (`src/xmvad/inspection/normal_twin.py`)
Answers the critical operator question: *"What nominal structure did the system expect this region to resemble?"*
- Traces prototype provenance to training normal samples (`prototype_id`, `category`, `sample_id`, `spatial_coord`).
- Queries top-$k$ normal references for representative defect patches (peak anomaly, peak topology, region medoid).
- Evaluates joint compatibility: $D_{\text{joint}}(j) = 0.5 \cdot d_{\text{rgb}}(j) + 0.5 \cdot d_{\text{xyz}}(j)$.
- Computes measured appearance difference ($\Delta E_{Lab}$) and signed surface deviation with local alignment confidence.

### Feature 2: Material Loss / Excess Volume Estimation (`src/xmvad/inspection/volume.py`)
Quantifies physical geometric deformation:
- Fits local reference surface (robust plane or quadratic surface) on surrounding normal ring points.
- Evaluates signed displacement field: $\delta_i = \text{signed\_distance}(\mathbf{p}_i, \text{reference\_surface})$ (positive = protruding, negative = recessed).
- Integrates volume using dual-mesh triangulated reference surface elements, completely eliminating artificial cliff-edge boundary artifacts.
- Reports missing volume, excess volume, and net signed volume separately.
- Achieves $< 0.1\%$ error on calibrated synthetic geometric benchmarks (circular depression, rectangular protrusion, Gaussian depression).

### Feature 3: Prototype Evidence Trace (`src/xmvad/inspection/prototype_trace.py`)
Exposes the core research mechanism of PNTC directly to researchers and operators:
- Displays top-$k$ RGB and XYZ prototypes with rank, distance, and training source sample provenance.
- Computes shared prototype IDs, Jaccard overlap, reciprocal rank overlap, and soft Jensen-Shannon divergence ($D_{\text{JS}}$).
- Formulates exact score decomposition:
  $$A_{\text{PNTC}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$$
- Reconstructs stored PNTC scores bit-for-bit.
- Generates evidence-grounded natural language rationale explaining why the patch was flagged.

### Feature 4: Uncertainty & Manual Review Guard (`src/xmvad/inspection/uncertainty.py` & `quality.py`)
Provides deployment-grade decision triage without altering canonical benchmark results:
- Decouples three distinct uncertainties: detection certainty, geometry measurement reliability, and sensor quality.
- Tracks sensor diagnostics: valid XYZ coverage, point density, RGB contrast, boundary proximity, surface fit residual, and retrieval margin gap ($d_{k+1} - d_k$).
- Recommends operational status:
  - `ACCEPT_NORMAL`
  - `DEFECT_DETECTED`
  - `MANUAL_REVIEW_RECOMMENDED`
- Flags deterministic warning codes (e.g. `LOW_XYZ_COVERAGE`, `NEAR_DECISION_THRESHOLD`, `OBJECT_BOUNDARY_REGION`).

---

## 4. Directory Structure

```
src/xmvad/inspection/
├── __init__.py          # Main inspect_sample() entry point
├── schema.py            # Complete data models and schemas
├── normal_twin.py       # Prototype provenance index & normal twin retrieval
├── prototype_trace.py   # Top-k retrieval, JSD, & PNTC score decomposition
├── volume.py            # Reference surface fitting & triangulated volume
├── quality.py           # Sensor coverage, boundary proximity, retrieval stability
├── uncertainty.py       # Decision margin & deployment Review Guard
├── report.py            # Unified ASCII industrial inspection report generator
└── visualize.py         # Multi-panel overlays, twin comparisons, & 3D HTML
```

---

## 5. Command-Line Tools

- **Single Sample Inspection**:
  ```bash
  python scripts/high_accuracy/inspect_sample.py \
      --sample cookie_combined_000 \
      --prediction results/raw_predictions/h5d_raw_predictions.npz \
      --output results/inspection/
  ```
- **Batch Dataset Inspection**:
  ```bash
  python scripts/high_accuracy/inspect_dataset.py \
      --predictions results/raw_predictions/h5d_raw_predictions.npz \
      --output_dir results/inspection/batch/ \
      --max_samples_per_category 5
  ```
- **Demo Case Bundles**:
  ```bash
  python scripts/high_accuracy/generate_demo_bundles.py
  ```
