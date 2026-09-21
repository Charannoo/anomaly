# PNTC AI Assistant — Conversational Inspection Explanation System

## Overview & Core Architecture

The **PNTC AI Assistant** is a post-inspection, natural-language explanation layer designed for industrial anomaly detection. It sits strictly downstream of the canonical, frozen **PNTC (Paired Neighborhood Topology Consistency)** detector and the **PNTC Inspection Suite**.

```
RGB Image  ───► DINOv2 ────┐
                            ├─► PNTC Core ─► PNTC Explain ─► PNTC Inspection Suite
XYZ Cloud  ───► Point-MAE ─┘                                        │
                                                                    ▼
                                                    VERIFIED STRUCTURED JSON
                                                                    │
                                                        ┌───────────┴───────────┐
                                                        ▼                       ▼
                                                    Gemini API              xAI Grok API
                                                        │                       │
                                                        └───────────┬───────────┘
                                                                    ▼
                                                            PNTC AI Assistant
                                                      (Conversational Explanation & Q&A)
```

---

## Strict Non-Detection Contract

> [!IMPORTANT]
> **The LLM does NOT detect anomalies, compute anomaly scores, or alter detection thresholds.**
> All anomaly scores, bounding contours, 3D surface deviations, volumetric quantities, prototype IDs, and topology metrics originate entirely from deterministic detector and physical measurement modules.

- **Detection**: Handled by PNTC (DINOv2 + Point-MAE paired neighborhood graph).
- **Explanation**: The assistant verbalizes, summarizes, and contextualizes the verified measurements.
- **Detector Invariance**: Canonical metrics remain permanently frozen:
  - **I-AUROC**: `0.96541000`
  - **P-AUROC**: `0.99416000`
  - **AUPRO@0.3**: `0.96939000`

---

## Provider Architecture & Official SDK Integration

The assistant uses an abstracted provider pattern (`LLMProvider`) located in [`src/xmvad/assistant/`](file:///c:/Users/CharanOp/xmv-ad/src/xmvad/assistant/):

1. **Google Gemini (`GeminiProvider`)**:
   - Primary default provider.
   - Integrated with the modern official `google-genai` SDK (`google.genai.Client`).
   - Automatic fallback to direct HTTP REST endpoint (`https://generativelanguage.googleapis.com/v1beta/models/...`) via `httpx`.
   - Default Model: `gemini-2.5-flash` (configurable via `PNTC_LLM_MODEL`).

2. **xAI Grok (`GrokProvider`)**:
   - Secondary / failover provider.
   - Integrated via official OpenAI-compatible REST completions endpoint (`https://api.x.ai/v1/chat/completions`) using `httpx`.
   - Default Model: `grok-2-latest` (configurable via `PNTC_LLM_MODEL`).

3. **Provider Factory & Failover (`provider_factory.py`)**:
   - Switch providers via `PNTC_LLM_PROVIDER=gemini` or `PNTC_LLM_PROVIDER=grok` without scattering conditionals across the codebase.
   - Transparent failover logging without secret exposure.

---

## Context Construction (`context_builder.py`)

The LLM never receives internal Python objects, raw point clouds, or feature tensors. Instead, [`build_assistant_context()`](file:///c:/Users/CharanOp/xmv-ad/src/xmvad/assistant/context_builder.py) produces a compact, purely factual JSON object containing:
- Sample metadata (ID, category)
- Inspection decision, score, and certainty
- Morphology & location (centroid, aspect ratio, orientation)
- Calibrated 3D geometry (maximum & mean depression/protrusion depth in mm)
- Calibrated volume estimation (missing & excess material in mm³)
- Evidence breakdown (RGB level, XYZ level, JS topology disagreement)
- Paired prototype reference & retrieval trace (shared top-5 IDs, JS divergence)
- Measurement quality indicators & reliability guards

---

## Response Modes

The assistant supports three distinct conversational styles via `ResponseMode`:
1. **`SIMPLE`**: Concise, plain-language summaries for machine operators and factory personnel.
2. **`TECHNICAL`**: Full quantitative engineering detail including signed depths, volumetric mm³, and cross-modal JS divergence metrics.
3. **`VIVA`**: Authoritative educational defense responses addressing viva panel questions about DINOv2, Point-MAE, topology consistency formulation, and canonical benchmark metrics using [`docs/PNTC_ASSISTANT_PROJECT_CONTEXT.json`](file:///c:/Users/CharanOp/xmv-ad/docs/PNTC_ASSISTANT_PROJECT_CONTEXT.json).

---

## UI Actions

Conversational questions can trigger frontend events via structured `UIAction` outputs:
- `"Show me the nearest normal example."` ➔ `SHOW_NORMAL_TWIN` (with `prototype_id`)
- `"Show 3D view"` ➔ `SHOW_3D_VIEW`
- `"Show prototype evidence"` ➔ `SHOW_PROTOTYPE_TRACE`
- `"Show heatmap"` ➔ `SHOW_HEATMAP`
- `"Highlight defect"` ➔ `HIGHLIGHT_DEFECT`
- `"Compare defects"` ➔ `COMPARE_DEFECTS`

---

## Deterministic Caching

To eliminate redundant API costs during repetitive inspections:
- Default explanations are cached using SHA-256:
  `hash(inspection_json + provider + model + prompt_version)`
- Re-querying identical frozen inspection results returns instant responses with `cached: true` and `latency_ms: 0.0`.
