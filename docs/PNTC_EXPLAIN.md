# PNTC Explain: Morphology-Aware Defect Characterization & Explanation

## Overview
**PNTC Explain** is an evidence-grounded post-detection characterization and explanation engine built on top of the frozen **Paired Neighborhood Topology Consistency (PNTC)** anomaly detection model.

While standard industrial anomaly detectors output scalar anomaly scores, 2D heatmaps, and binary pass/fail decisions, PNTC Explain translates these continuous multimodal outputs into structured, interpretable defect descriptions detailing:
- **Spatial Location**: Where the defect is located relative to the object's physical surface.
- **2D Morphology & Orientation**: The 2D geometric shape (area, aspect ratio, circularity, solidity, compactness, axis orientation).
- **3D Physical Dimensions**: Calibrated metric dimensions ($\text{mm}$ width, height, bounding depth) and 3D surface area ($\text{mm}^2$).
- **Local Reference Surface & Signed Deviations**: Robust plane fitting on surrounding normal surfaces to determine whether a defect is recessed (depression), raised (protrusion), or irregularly warped.
- **Visual Appearance Deviations**: Color shift ($\text{CIE Lab } \Delta E$), texture variance, and luminance contrast relative to the surrounding normal ring.
- **Cross-Modal Detection Evidence**: Explicit breakdown of RGB evidence, 3D evidence, and Paired Prototype Topology Disagreement $T(p)$.
- **Deterministic Natural Language Rationale**: Factual, evidence-grounded explanations with zero LLM hallucination risk.

---

## 1. Input Contract & Execution Architecture

```
[Registered RGB Image (224x224)]   [Registered XYZ Point Cloud (224x224x3)]
               \                                   /
                \                                 /
              [Frozen PNTC Inference Engine (H5-D)]
                                |
             +------------------+------------------+
             |                                     |
    [Image Anomaly Score]               [2D Anomaly Heatmap (224x224)]
             |                                     |
             +------------------+------------------+
                                |
                                v
                      [PNTC Explain Module]
                                |
       +------------------------+------------------------+
       |                        |                        |
[Connected Component      [Local Reference        [Cross-Modal Evidence
 Region Extraction]        Surface Fitting]        & Topology Analysis]
       |                        |                        |
       +------------------------+------------------------+
                                |
                                v
               [Structured JSON & Technical Report]
                                |
        +-----------------------+-----------------------+
        |                                               |
[Annotated 2D Overlay PNG]                 [Interactive 3D HTML Viewer]
```

### Python API Usage
```python
from xmvad.explain import explain_anomaly, ExplainInput

# Construct input contract
inp = ExplainInput(
    rgb=rgb_image,                      # (H, W, 3) uint8 or float [0, 1]
    xyz=xyz_coordinates,                # (H, W, 3) float in mm
    anomaly_map=pntc_anomaly_map,       # (H, W) continuous anomaly heatmap
    image_score=image_score,            # Scalar anomaly score
    category="cookie",
    sample_id="cookie_001",
    coordinate_unit="mm",               # 'mm', 'm', or 'unknown'
    threshold=0.50                      # Operating threshold
)

# Generate structured explanation
report = explain_anomaly(inp)

print("Decision:", report.decision)
print("Summary:", report.summary_explanation)
print("Defects detected:", report.num_defects)
```

---

## 2. Measurement & Characterization Algorithms

### 2.1 Coordinate Unit Management & Physical Scaling
- **Unit Verification**: Coordinates are inspected for declared units (`"mm"`, `"m"`, or `"unknown"`).
- **Safety Policy**: If units are uncalibrated or normalized, physical metric dimensions ($\text{mm}, \text{mm}^2$) are strictly suppressed. The system falls back to pixel area, bounding dimensions, and percentage of foreground object surface.

### 2.2 Connected Region Extraction & Importance Ranking
1. **Calibrated Thresholding**: Continuous anomaly scores are binarized using a train-normal calibrated operating threshold (default $0.50$).
2. **Morphological Noise Cleaning**: Morphological opening with an 8-connectivity structuring element removes isolated 1–2 pixel noise spikes.
3. **Integrated Importance Ranking**: Connected components are ranked by integrated anomaly evidence:
   $$\text{Importance}(k) = \bar{S}_k \times \sqrt{\text{Area}_k}$$
   where $\bar{S}_k$ is the mean anomaly score in region $k$.

### 2.3 2D Morphology & Shape Descriptors
- **Aspect Ratio**: $\text{Major Axis} / \text{Minor Axis}$ derived from image central moments ($\mu_{20}, \mu_{02}, \mu_{11}$).
- **Circularity**: $4\pi A / P^2 \in [0, 1]$, where $P$ is the morphological perimeter.
- **Solidity**: $A / A_{\text{convex\_hull}} \in [0, 1]$.
- **Extent**: $A / (W_{\text{bbox}} \cdot H_{\text{bbox}})$.
- **Conservative Shape Labels**: Derived from metric combinations:
  - `"compact circular region"` ($\text{Circularity} \ge 0.70, \text{Aspect Ratio} \le 1.4$)
  - `"thin elongated stripe"` ($\text{Aspect Ratio} \ge 3.5, \text{Solidity} \ge 0.75$)
  - `"elongated irregular region"` ($\text{Aspect Ratio} \ge 2.0, \text{Solidity} < 0.80$)
  - `"compact irregular region"` ($\text{Solidity} \ge 0.80, \text{Extent} \ge 0.55$)

### 2.4 Local Reference Surface & Signed Deviations
1. **Surrounding Normal Ring**: A morphological ring surrounding the defect is extracted by dilating the defect mask and filtering out invalid points and anomalous pixels.
2. **Robust Plane Fitting**: A local reference plane $\mathbf{n} \cdot \mathbf{p} + d = 0$ is fitted to ring points using SVD with iterative outlier rejection. The normal $\mathbf{n}$ is oriented outward toward the camera.
3. **Signed Surface Deviations**: For each valid 3D point $\mathbf{p}_i$ in the defect region, the signed distance is:
   $$\delta_i = \mathbf{n} \cdot \mathbf{p}_i + d$$
   - $\delta_i > 0$: Point extends outward toward camera (**raised protrusion**).
   - $\delta_i < 0$: Point extends inward away from camera (**recessed depression**).
4. **Structural Classification**:
   - `"localized depression"` ($\bar{\delta} < -1.8 \sigma_{\text{ring}}$ and $\text{max depression} > 1.8 \times \text{max protrusion}$)
   - `"raised protrusion"` ($\bar{\delta} > +1.8 \sigma_{\text{ring}}$ and $\text{max protrusion} > 1.8 \times \text{max depression}$)
   - `"irregular local deformation"` ($\text{RMS} > 2.0 \sigma_{\text{ring}}$ with mixed positive and negative deviations)
   - `"primarily appearance-based anomaly"` ($|\bar{\delta}| < 1.5 \sigma_{\text{ring}}$)

### 2.5 Visual Appearance Contrast
- **Mean Color Shift**: Euclidean RGB difference relative to surrounding normal ring.
- **CIE Lab $\Delta E$**: Perceptual color difference calculated using D65 reference white.
- **Luminance Contrast**: Weber/Michelson contrast ratio between defect and ring.
- **Texture Variance**: Local gray-level intensity variance difference.

### 2.6 Cross-Modal Evidence & Topology Disagreement
PNTC evidence components are extracted and mapped into standardized qualitative levels:
- **RGB Evidence**: `Low` ($<0.35$), `Moderate` ($0.35\text{--}0.60$), `High` ($0.60\text{--}0.82$), `Very High` ($\ge 0.82$).
- **3D Evidence**: Standardized using the Point-MAE geometric distance.
- **Topology Disagreement**: Normalized Jensen-Shannon Divergence $T(p) = \text{JSD} / \ln 2$.

---

## 3. Structured JSON Schema Example

```json
{
  "sample_id": "cookie_combined_000",
  "category": "cookie",
  "decision": "anomalous",
  "image_anomaly_score": 0.65715,
  "detection_confidence_percent": 79.4,
  "operating_threshold": 0.5,
  "coordinate_unit": "mm",
  "physical_measurements_available": true,
  "num_defects": 1,
  "defects": [
    {
      "id": 1,
      "location": {
        "label": "central surface",
        "centroid_px": [145, 79],
        "centroid_normalized": [0.6498, 0.3534],
        "relative_to_object": true
      },
      "size": {
        "width_px": 19,
        "height_px": 20,
        "area_px": 303,
        "major_length_mm": 9.69,
        "minor_length_mm": 8.12,
        "bounding_depth_mm": 0.43,
        "projected_area_mm2": 61.85,
        "surface_area_3d_mm2": 68.38,
        "physical_units_available": true
      },
      "shape": {
        "label": "compact circular region",
        "aspect_ratio": 1.24,
        "circularity": 1.0,
        "solidity": 1.0,
        "orientation_deg": 49.4
      },
      "geometry": {
        "label": "rough raised protrusion",
        "mean_deviation": 0.446,
        "max_protrusion_mm": 0.68,
        "rms_deviation": 0.463,
        "surface_z_score": 8.93,
        "roughness_ratio": 2.97,
        "measurement_confidence": 0.97,
        "unit": "mm"
      },
      "appearance": {
        "deviation_label": "strong",
        "delta_e_lab": 27.52,
        "contrast_ratio": 1.27
      },
      "evidence": {
        "rgb_level": "Moderate",
        "xyz_level": "Moderate",
        "topology_level": "High",
        "gate_level": "High"
      },
      "explanation": "A compact circular region was detected in the central surface, measuring approximately 9.69 mm × 8.12 mm. Visual inspection reveals strong appearance deviation from the normal reference ring (CIE Lab ΔE = 27.52). Both visual and geometric branches contribute consistent anomaly evidence (RGB: Moderate, 3D: Moderate, Topology: High)."
    }
  ]
}
```

---

## 4. Limitations & Scope

1. **Physical Unit Calibration**: Physical $\text{mm}$ dimensions require registered, calibrated XYZ sensor data. Uncalibrated inputs default to pixel and percentage measurements.
2. **Evidence-Based Characterization vs Semantic Cause**: The module characterizes *measured morphology* (e.g. `"localized depression"`, `"elongated stripe"`). It does not infer manufacturing root causes (e.g. `"tooling chatter"`, `"corrosion"`) without separate domain classifiers.
3. **Reference Surface Assumptions**: Local plane fitting assumes local surface planarity on the reference ring. Complex high-frequency curved geometries may exhibit lower measurement confidence.
4. **Detector Invariance**: PNTC Explain is strictly a post-detection layer. It does not alter anomaly scores or benchmark evaluation metrics.
