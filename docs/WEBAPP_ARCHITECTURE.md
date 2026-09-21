# PNTC Inspect — Web Application Architecture

## 1. System Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NEXT.JS FRONTEND (CLIENT)                       │
│                                                                        │
│  React 19 + TypeScript + Tailwind CSS + Lucide + Recharts + Plotly     │
│  State: Server State via TanStack Query / Fetch + Local React UI State  │
│  Routing: Next.js App Router (/dashboard, /inspect, /inspections, ...) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST / SSE / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND (SERVER)                        │
│                                                                        │
│  Engine: Python 3.13 + Uvicorn Async Server (Port 8000)                │
│  Endpoints:                                                            │
│    - /api/inspections (List, Run, Get, Delete)                         │
│    - /api/inspections/{id}/status (SSE real-time pipeline stages)      │
│    - /api/inspections/{id}/artifacts/{name} (Images, 3D HTML, JSON)     │
│    - /api/assistant/chat & /api/assistant/explain                      │
│    - /api/demo/samples (Real MVTec-3D benchmark cases)                 │
│    - /api/model/info (Read-only architecture parameters & benchmarks)  │
│  Persistence: SQLite (data/inspections.db) for metadata & logs        │
└───────┬───────────────────────────┬───────────────────────────┬────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌─────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  PNTC DETECTOR  │       │  INSPECTION SUITE│       │   AI ASSISTANT   │
│  (FROZEN MODEL) │       │  3D, Volume,     │       │  Gemini & Grok   │
│  DINOv2 + Pt-MAE│       │  Normal Twin     │       │  Grounding Check │
└─────────────────┘       └──────────────────┘       └──────────────────┘
```

---

## 2. API Endpoints Specification

### A. Inspections API
- `GET /api/inspections`: Returns paginated inspection history with filters (`category`, `status`, `review_recommended`, `search`).
- `POST /api/inspections`: Accepts multipart RGB image + XYZ point cloud (or `.npz` container) or `demo_case_id`. Runs or queues inspection.
- `GET /api/inspections/{id}`: Returns complete verified `InspectionReport` JSON.
- `GET /api/inspections/{id}/status`: Server-Sent Events (SSE) streaming the pipeline execution stages:
  1. `Input Validation`
  2. `RGB Feature Extraction`
  3. `XYZ Feature Extraction`
  4. `Normal Prototype Retrieval`
  5. `PNTC Scoring`
  6. `Defect Segmentation`
  7. `Geometry Analysis`
  8. `Normal Twin Retrieval`
  9. `Evidence Trace`
  10. `Report Generation`
  11. `AI Explanation`
- `GET /api/inspections/{id}/artifacts/{artifact_type}`: Streams binary PNGs (`rgb`, `depth`, `heatmap`, `overlay`, `normal_twin`, `prototype_trace`, `surface_difference`) or Plotly 3D HTML.
- `GET /api/inspections/{id}/export/{format}`: Downloads report as JSON, Markdown, or styled printable HTML/PDF.

### B. Assistant API
- `POST /api/assistant/chat`: Submits user question with `sample_id`, `conversation_id`, and `response_mode` (`SIMPLE`, `TECHNICAL`, `VIVA`).
- `POST /api/assistant/explain`: Fetches default grounded inspection summary with SHA-256 caching.
- `GET /api/assistant/provider`: Returns provider status (`gemini` / `grok`) without exposing credentials.

### C. Model & Research API
- `GET /api/model/info`: Returns read-only architecture parameters (DINOv2 ViT-B/14, Point-MAE, $k=5$, $\lambda=0.35$, coreset size 15,000) and canonical verification metrics.
- `GET /api/research/ablations`: Returns official ablation table (H4B through H5-D) and reproducibility hashes.
- `GET /api/demo/cases`: Lists the 6 verified real industrial demo cases from `results/demo_cases/`.

---

## 3. Database Persistence (`data/inspections.db`)

Lightweight SQLite schema:
```sql
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL,          -- 'NORMAL', 'DEFECT_DETECTED', 'MANUAL_REVIEW'
    decision TEXT NOT NULL,        -- 'normal', 'anomalous'
    anomaly_score REAL NOT NULL,
    operating_threshold REAL NOT NULL,
    num_defects INTEGER NOT NULL,
    decision_certainty TEXT NOT NULL,
    measurement_reliability TEXT NOT NULL,
    manual_review_recommended INTEGER NOT NULL,
    execution_time_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_json TEXT NOT NULL,       -- Complete structured InspectionReport JSON
    artifacts_dir TEXT NOT NULL
);
```
Large binary arrays (point clouds, features) remain on disk; only references and JSON metrics are stored in SQLite.
