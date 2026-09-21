# PNTC Inspect — Web Application Repository & Architecture Audit

## 1. Executive Summary

This document audits the entire `xmv-ad` repository to prepare for the development of **PNTC Inspect** (*Multimodal RGB–3D Industrial Inspection*), a web application built in the aesthetic of precision industrial instrumentation (Siemens, Hexagon, Keyence, Cognex, ZEISS).

The core PNTC detector and its canonical benchmark metrics are **strictly frozen**:
- **I-AUROC**: `0.96541000`
- **P-AUROC**: `0.99416000`
- **AUPRO@0.3**: `0.96939000`
- **Frozen Tag**: `h5d-pntc-verified`

---

## 2. Existing Assets & Capabilities

### A. Core Detection & Scoring (`src/xmvad/`)
- **Backbones**: DINOv2 (ViT-S/14 or ViT-B/14 for RGB) + Point-MAE (unstructured XYZ point clouds).
- **Core Detector**: `PNTC` (Paired Neighborhood Topology Consistency via Jensen-Shannon neighborhood divergence and cross-modal anomaly gating).
- **Frozen Predictions**: Canonical evaluation results stored in `results/H5_raw_metric_verification.csv`, `results/raw_predictions/h5d_raw_predictions.npz`, and `results/h5d_raw_predictions_replay.npz`.

### B. PNTC Explain (`src/xmvad/explain/`)
- **Morphology 2D**: Defect bounding boxes, aspect ratio, circularity, orientation, solidity, projected area.
- **Physical 3D Sizing**: Calibrated millimeter major/minor lengths and surface areas.
- **Surface Diagnostics**: Signed depth deviation, localized depression, raised protrusion, appearance-only blemish.
- **Reasoning & Evidence**: Dual-modality contribution (`RGB_DOMINANT`, `DEPTH_DOMINANT`, `JOINT_CONSISTENT`, `TOPOLOGY_CONFLICT`).

### C. PNTC Inspection Suite (`src/xmvad/inspection/`)
- **Normal Twin Retrieval (`normal_twin.py`)**: Deterministic nearest paired normal reference prototype lookup (`prototype_id`, `training_sample_id`, `joint_score`, `rgb_distance`, `xyz_distance`).
- **Prototype Evidence Trace (`prototype_trace.py`)**: Top-5 RGB and XYZ prototype neighbors, Jaccard similarity, JS divergence, gate value, base vs topology decomposition.
- **Volume Quantification (`volume.py`)**: Missing material volume (depression mm³), excess material volume (protrusion mm³), robust planar/polynomial reference fit.
- **Uncertainty & Review Guard (`uncertainty.py`)**: Automated recommendation for human manual inspection (`ACCEPT`, `DEFECT_DETECTED`, `MANUAL_REVIEW_RECOMMENDED`).
- **Visualization Engine (`visualize.py`)**: Generates 2D overlay plots, normal twin side-by-side comparisons, prototype trace rank plots, and interactive Plotly WebGL 3D HTML models.

### D. AI Assistant Subsystem (`src/xmvad/assistant/`)
- **Dual Providers**: Google Gemini (via official `google-genai` SDK + REST fallback) and xAI Grok (via REST).
- **Grounding Guardrails (`guardrails.py`)**: Strict number verification against context, prompt injection defense, causal claim rejection.
- **Session & History (`history.py`)**: Thread-safe multi-turn session tracking with defect reference disambiguation.
- **Caching (`chat.py`)**: Deterministic SHA-256 caching of default inspection explanations.
- **Project Context (`docs/PNTC_ASSISTANT_PROJECT_CONTEXT.json`)**: Verified project facts for Viva/defense mode.

### E. Prepared Benchmark & Demo Data (`results/demo_cases/`)
Six verified industrial inspection cases with full artifacts:
1. `01_strong_defect`: Cookie combined defect (15.8 × 15.9 mm, 3.37 mm depression, 620.4 mm³ missing volume, Prototype #3791).
2. `02_primarily_rgb_defect`: Surface discoloration / contamination without major geometric deformation.
3. `03_primarily_geometric_defect`: Subtle physical dent / depression with minimal color change.
4. `04_strong_topology_disagreement`: Severe cross-modal divergence triggering high PNTC penalty.
5. `05_borderline_manual_review`: Low confidence / near-threshold case triggering Review Guard.
6. `06_failure_limitation_boundary`: Edge / boundary artifact demonstrating transparency and limitations.

Each folder contains:
- `inspection.json` (Structured output adhering to `InspectionReport` schema)
- `original_rgb.png`, `depth_or_xyz.png`, `pntc_heatmap.png`, `annotated_defects.png`
- `normal_twin.png`, `prototype_trace.png`, `surface_difference.png`
- `3d_view.html` (Interactive WebGL 3D model)
- `inspection_report.txt`

---

## 3. What Needs to Be Created

### A. Modern Industrial Web Frontend (`web/` or Next.js App)
A standalone, high-performance web application utilizing:
- **Framework**: Next.js 14+ (App Router) with React 19, TypeScript.
- **Styling**: Tailwind CSS configured with the strict dark industrial palette (`#0B0D10` bg, `#15191F` panel, `#5BB8C4` accent).
- **Icons**: Lucide Icons.
- **Typography**: Inter / Geist with `font-variant-numeric: tabular-nums`.
- **Aesthetic**: Restrained, scientific equipment UI (Siemens/ZEISS style, no glowing gradients, no AI clichés).

### B. High-Performance FastAPI Backend (`src/xmvad/api/`)
A unified FastAPI backend replacing ad-hoc scripts with robust REST & SSE endpoints:
- Inspection management and sample loading.
- Real-time pipeline orchestration with simulated/actual stage progress (Input Validation → Feature Extraction → Normal Prototype Retrieval → PNTC Scoring → Defect Segmentation → Geometry Analysis → Normal Twin Retrieval → Evidence Trace → AI Explanation).
- SQLite persistence (`data/inspections.db`) for inspection history, review states, and analytics.
- 3D point cloud & mesh streaming endpoint.
- Direct integration with `PNTCAssistant` (`/api/assistant/chat`, `/api/assistant/explain`).
- PDF and JSON technical report export.

---

## 4. Architectural Dataflow

```
   ┌─────────────────────────────────────────────────────────────┐
   │                   NEXT.JS FRONTEND (PORT 3000)              │
   │  Dashboard | New Inspect | Result | 3D View | History | ... │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ HTTP / SSE / REST
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                   FASTAPI BACKEND (PORT 8000)               │
   │   /api/inspections  /api/demo  /api/assistant  /api/model   │
   └───────┬──────────────────────┬──────────────────────┬───────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│   PNTC PIPELINE    │ │   SQLITE STORAGE   │ │    AI ASSISTANT    │
│  Explain + Suite   │ │  inspections.db    │ │   Gemini + Grok    │
└────────────────────┘ └────────────────────┘ └────────────────────┘
```

---

## 5. Non-Regression Guarantee
All code changes will remain strictly external to the core detector. The verification test `tests/inspection/test_regression_pntc_metrics.py` will be run throughout to ensure zero drift from the frozen metrics:
- **I-AUROC = 0.96541000**
- **P-AUROC = 0.99416000**
- **AUPRO@0.3 = 0.96939000**
