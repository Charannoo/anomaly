# XMV-AD: Lightweight Modality-Preserving RGB-D Fusion for Explainable Industrial Anomaly Detection

> **Status: PHASE 5 COMPLETE — Paper-protocol hardening + modality-aware**
> **explainability on the frozen E4 residual fusion.**
>
> - Final model identity: **"XMV-AD Residual Fusion / E4"** — frozen Phase 2
>   RGB + Z-depth students fused by a tiny zero-init residual concat block
>   @1/16 (8,738 trainable params). See `docs/PHASE4A_CONCAT_FUSION.md`,
>   `docs/PHASE5.md`.
> - Honest ablation: E5/E6/E7 cross-attention does **NOT** beat E4 (pooled
>   image-level -0.040; AUPRO .73 vs .85). E7 is reported as the published
>   **"XMV-Attention Ablation"**. See `docs/PHASE4B_ATTENTION_ABLATION.md`.
> - Base-paper numbers (MVTec 3D-AD 2022) and the M3DM comparator are in
>   `docs/BASE_PAPER_2022.md` + `experiments/tables/paper_baseline_table.csv`.
> - Controlled modality-attribution benchmark (train-normal only): dominant
>   modality accuracy .781, macro-F1 .873 (corrected, see below), AttributionConsistency .848,
>   fused spatial AUROC ~.96. Safety: relative evidence, never causal.
> - Macro-F1 correction: the Phase-5 script originally reported .918 from a
>   bug (JOINT always scored correct); corrected to .873 in the Phase-6 audit
>   (`experiments/tables/attribution_metric_audit.md`). No model or thresholds
>   changed.
> - Phase 6 deployment: **ONNX Runtime FP32 recommended** (120 FPS, e2e p95
>   ~12.9 ms, 2.3 MB, exact metric parity with frozen model). INT8 PTQ failed
>   pre-declared accuracy thresholds and is reported as such. See
>   `docs/PHASE6_DEPLOYMENT.md` + `experiments/tables/deployment_final.md`
>   and the Pareto plot `experiments/tables/deployment_pareto.png`.

## 1. Project summary

Factories need automatic visual inspection for defects such as cracks, scratches,
dents, bumps, discoloration, deformation, missing material and surface
irregularities. RGB cameras capture appearance defects; depth/3D sensors capture
geometric defects. XMV-AD investigates whether a **lightweight RGB-D anomaly
detector** with **direct bidirectional cross-modal interaction** can deliver
useful detection + localization **while retaining separate RGB and depth
evidence**, **validating modality attribution**, and **running on CPU**.

Three objectives are optimized jointly:

- **A.** anomaly detection / localization accuracy
- **B.** computational efficiency (CPU-deployable)
- **C.** modality-aware explainability (which sensor contributed, and is that claim faithful?)

## 2. Research motivation

Multimodal industrial anomaly detection (e.g. M3DM, CVPR 2023) has shown RGB and
3D are complementary, but typical designs rely on heavy pretrained image +
point-cloud backbones, alignment, fusion, and multiple memory banks.

XMV-AD explores a different design philosophy:

```
M3DM:  heavy RGB + point-cloud reps -> alignment -> fusion -> memory banks -> decision
XMV-AD: light RGB encoder <-> frozen students + tiny residual concat @1/16
        -> modality-preserving heads -> RGB map + Depth map -> calibrated fusion
        -> final map -> validated modality attribution
```

Intended contribution (*tested*, see `experiments/tables/`):

> A compact modality-preserving RGB-Z anomaly detector using a zero-init
> residual feature concatenation that enables efficient localization while
> retaining and quantitatively validating sensor-specific RGB/depth evidence.

## 3. Base paper

Primary baseline: **M3DM / Multi-3D-Memory** — Yue Wang et al.,
"Multimodal Industrial Anomaly Detection via Hybrid Fusion", CVPR 2023.
See `docs/BASELINES.md`. Shared metrics: Image-AUROC, Pixel-AUROC, AUPRO/PRO.

Fairness note: M3DM uses RGB + full XYZ point-cloud features; XMV-AD v1 uses
RGB + aligned Z-depth. They do **not** consume identical representations
(see `docs/BASELINES.md`).

## 4. Architecture

Final identity (E4 "Residual Fusion"); E5/E6/E7 cross-attention was a tested
ablation, not the final model:

```
RGB ──> frozen student ─┐                 ┌──> A_RGB
                        ├─ concat+resid ──┤
Depth ─> frozen student ┘  (gamma gates)  └──> A_DEPTH ──> calibrated fusion ──> A_FINAL ──> attribution
```

Details: `docs/ARCHITECTURE.md`, `docs/PHASE4A_CONCAT_FUSION.md`. Phase 2
students (encoders + heads) are frozen; the only trainable part of E4 is the
ConcatFusion block (8,738 params).

## 5. Installation

```powershell
cd xmv-ad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Requires Python >= 3.11. Inference targets CPU-only; training may use CUDA
when available but must never require it.

## 6. Dataset preparation

**Primary benchmark: MVTec 3D-AD** (RGB + organized XYZ, normal-only train).

MVTec 3D-AD is **not** included in this repository.

```
Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD
dataset page and extract/copy it to data/raw/mvtec3d/.
```

Do NOT use Kaggle mirrors unless explicitly approved.

Expected layout (`docs/DATASET.md`):

```
data/raw/mvtec3d/<category>/{train,validation,test}/...
```

Validate with:

```powershell
python scripts/inspect_dataset.py --data data/raw/mvtec3d
python scripts/visualize_sample.py --data data/raw/mvtec3d --out experiments/figures/data_checks --num-per-category 25
```

## 7. Preprocessing

```powershell
python scripts/preprocess_mvtec3d.py --input data/raw/mvtec3d --output data/processed/mvtec3d
```

Never modifies `data/raw`. Depth stats are fit on **train-normal only**.

## 8–12. Baselines / training / evaluation / benchmarking / explainability

| Stage | Command | Status |
|---|---|---|
| RGB baseline | `python scripts/evaluate_concat.py --config configs/baseline_rgb.yaml` | Done (E1) |
| Depth baseline | `python scripts/evaluate_concat.py --config configs/baseline_depth.yaml` | Done (E2) |
| Late/concat fusion | `scripts/evaluate_concat.py` + `docs/PHASE3*.md` | Done (E3A/E3C) |
| XMV-AD (Residual Fusion) | `scripts/train_concat.py --config configs/concat_fusion.yaml` | Done (E4) |
| Cross-attention ablation | `scripts/train_xmv.py` (E5/E6/E7) | Ablated - see Phase 4B docs |
| Evaluation | `scripts/evaluate_concat.py`, `scripts/eval_e4_native_resolution.py` | Done |
| Official metrics | `src/xmvad/metrics/` (pro.py official protocol) | Done |
| CPU benchmark | `scripts/benchmark_*.py` (E5E6E7_dev_latency.csv) | Done |
| Explainability eval | `scripts/generate_attribution_data.py`, `eval_attribution.py` | Done (Phase 5) |
| Full reproduction | `bash scripts/reproduce_all.sh` | after dataset + weights present |

## 13. Results

Headline numbers on **MVTec 3D-AD**, evaluated at native GT resolution with
the official MVTec metric protocol. Every row is labelled with provenance.
See `experiments/tables/paper_baseline_table.csv` and
`docs/TABLES_PAPER_BASELINE.md`.

| Method | I-AUROC | P-AUROC | AUPRO | Fusion params | CPU p95 (ms) | Provenance |
|---|---|---|---|---|---|---|
| Voxel f-AnoGAN (3D+RGB) | 0.517 | — | 0.639 | — | — | Reported by prior paper (MVTec 3D-AD 2022) |
| Voxel Variation Model (3D-only) | 0.699 | — | 0.492 | — | — | Reported by prior paper (MVTec 3D-AD 2022) |
| M3DM (RGB+3D point cloud) | 0.945 | — | 0.964 | — | — | Reported by prior paper (CVPR 2023); different input representation |
| E1 RGB student | 0.6401 | 0.8950 | 0.6855 | 959,097 (full branch) | 14.8 | Measured by us |
| E2 Depth student | 0.5340 | 0.7542 | 0.4918 | 959,097 (full branch) | 15.0 | Measured by us |
| E3A late fusion (score mean) | 0.6400 | 0.9148 | 0.7224 | 0 | 30.3 | Measured by us |
| **E4 Residual Fusion (XMV-AD)** | **0.6536** | **0.9582** | **0.8572** | 8,738 | 34.0 | Measured by us |
| E7 Attention Ablation | 0.6337 | 0.9179 | 0.7312 | 12,770 | 38.0 | Measured by us (fails \| E4) |

Per-category numbers, native-vs-256 deltas and full baseline tables in
`experiments/tables/E4_official_metric_verification.csv`,
`E4_concat_metrics_native.csv`.

Modality-aware explainability (Phase 5, train-normal-only controlled
benchmark, `experiments/tables/attribution_metrics.md`):

| Metric | Value |
|---|---|
| Dominant-modality accuracy (RGB/DEPTH/JOINT) | 0.781 |
| Macro-F1 (3 injected modalities) | 0.873 |
| AttributionConsistency (evidence \| decision-ablation direction) | 0.848 |
| Fused pixel-AUROC vs injected mask (pooled) | ~0.96 |

Real-defect evidence mix (1,197 test defects; region = GT): 35.4% RGB-dominant,
4.9% depth-dominant, 38.9% joint, 20.8% uncertain (evidence is *relative*, not
causal). Failure/discovery candidates and operator figures:
`experiments/figures/explainability_failures/`.

## 14. Reproduction instructions

Phase 1 reproduction:

```powershell
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/inspect_dataset.py --data data/raw/mvtec3d
python scripts/visualize_sample.py --data data/raw/mvtec3d --out experiments/figures/data_checks
```

Later phases: `bash scripts/reproduce_all.sh` (requires dataset + weights).

## 15. Citation

```bibtex
@misc{xmvad2026,
  title  = {XMV-AD: Lightweight Modality-Preserving RGB-D Fusion for Explainable Industrial Anomaly Detection},
  author = {XMV-AD Contributors},
  year   = {2026},
  note   = {Research implementation; E4 residual fusion + validated modality attribution}
}
```

Base comparison:

```bibtex
@inproceedings{wang2023m3dm,
  title     = {Multimodal Industrial Anomaly Detection via Hybrid Fusion},
  author    = {Wang, Yue and et al.},
  booktitle = {CVPR},
  year      = {2023}
}
```

## Research integrity

- No fabricated data, results, or metric values.
- No tuning on test; normal-only training.
- Attention maps are interaction visualizations, not causal proof.
- Report per-category metrics incl. failures; disclose RGB+Z vs RGB+XYZ gap.
- See `docs/RESEARCH_PLAN.md`.
