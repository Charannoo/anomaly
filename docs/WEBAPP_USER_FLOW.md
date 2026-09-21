# PNTC Inspect — Web Application User Flow & Interaction Specification

## 1. Primary Operational Flows

```
[1. Dashboard] ──────────► [2. New Inspection] ──────────► [3. Processing Pipeline]
      ▲                            │ (Upload / Demo Sample)           │ (Real SSE Stages)
      │                            ▼                                  ▼
[6. History & Compare] ◄── [5. Interactive 3D Modal] ◄─── [4. Inspection Result View]
                                                              ├── Visual Inspection (Tabs)
                                                              ├── Defect Inspector (Measurements)
                                                              └── Right Drawer:
                                                                  ├── Grounded Assistant Q&A
                                                                  ├── Normal Twin Comparison
                                                                  ├── Prototype Trace
                                                                  └── Measurement Quality
```

---

## 2. Step-by-Step Flow Details

### Flow 1: New Inspection & Execution (`/inspect`)
1. **Navigation**: User clicks **"New Inspection"** from Top Bar or Sidebar.
2. **Input Selection**:
   - **Option A (Manual Upload)**: User drops an RGB image (left pane) and a corresponding XYZ point cloud / depth map (right pane). Real-time client & backend validation verifies image dimensions, spatial alignment, and XYZ validity fraction.
   - **Option B (Demo Sample Library)**: User clicks **"Load Demo Sample"**, opening a drawer displaying 6 real industrial MVTec-3D benchmark cases with preview thumbnails and defect categories.
3. **Execution Options**: User toggles auxiliary features (Generate 3D view, Generate AI explanation, Assistant mode).
4. **Execution**: User clicks **"Run Inspection"**.

### Flow 2: Live Processing Pipeline (`/inspect/[id]/processing`)
1. Displays real-time progress through Server-Sent Events (SSE).
2. Central architectural schematic highlights active computation:
   - `Input Validation` ➔ `Feature Extraction` ➔ `Memory Retrieval` ➔ `PNTC Scoring` ➔ `Defect Segmentation` ➔ `Geometry Analysis` ➔ `Normal Twin` ➔ `Evidence Trace` ➔ `Report Generation`.
3. Auto-redirects to `/inspect/[id]` upon completion.

### Flow 3: Comprehensive Result Review (`/inspect/[id]`)
1. **Status Banner**: Displays decision (`DEFECT DETECTED`, `NORMAL`, or `MANUAL REVIEW RECOMMENDED`), score vs. threshold, and certainty.
2. **Left Panel (Visual Inspection)**: User switches between views:
   - `Overlay`: Shows RGB + heatmaps + defect bounding boxes (`#1`, `#2`) with hover tooltip.
   - `RGB`: Raw input sensor capture.
   - `Anomaly Map`: Calibrated inferno heatmap.
   - `XYZ / Depth`: Sensor depth scan.
   - `Normal Twin`: Side-by-side comparison with retrieved prototype analogue.
   - `Difference`: Nominal-minus-actual surface difference map.
3. **Center Panel (Defect Inspector)**: Displays tabular micro-measurements:
   - Morphology (major/minor lengths, aspect ratio, circularity, area in mm²).
   - Calibrated 3D Geometry (max & mean depression/protrusion depth in mm).
   - Volumetric Loss (missing mm³, excess mm³, reference plane residual).
   - Cross-Modal Evidence breakdown (RGB level, XYZ level, JS topology disagreement).
4. **Right Panel (Inspector Drawer)**:
   - **Tab 1: Explain (AI Assistant)**: Compact conversational chat grounded strictly in verified inspection metrics. Includes quick queries (*"Why was this flagged?"*, *"How large is the defect?"*, *"Why do RGB and XYZ disagree?"*).
   - **Tab 2: Normal Twin**: Detailed breakdown of retrieved nominal prototype twin, source sample, distance metrics, and alignment certainty.
   - **Tab 3: Evidence Trace**: Top-5 RGB vs. Top-5 XYZ retrieval ranks, Jaccard index, JS divergence, gate value, and mathematical PNTC score decomposition equation.
   - **Tab 4: Quality**: Sensor point validity fraction, surface fit reliability, and Review Guard recommendation.

### Flow 4: Interactive 3D Metrology (`/inspect/[id]/3d` or Viewport)
1. User clicks **"3D View"** or **"Open 3D Defect"**.
2. Renders an interactive Plotly / WebGL 3D point cloud and reconstructed mesh.
3. Allows 360° rotation, pan, zoom, reference plane toggling, and visual color-coding of physical surface depression depth.

### Flow 5: Report Export
1. User clicks **"Export Report"** in top-right toolbar.
2. Formats available:
   - **Printable Technical Report (HTML/PDF)**: Complete ISO-style inspection certificate including calibration timestamps, sensor evidence, defect crops, and sign-off block.
   - **Machine-Readable JSON**: Complete structured data adhering to `InspectionReport` schema.
