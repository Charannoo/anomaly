# XMV-AD-H: High-Accuracy Research Track — Master Plan

> Goal (user-specified): build a **high-accuracy** MVTec-3D AD detector that
> *beats the original M3DM on the official metrics* (I-AUROC, P-AUROC, AUPRO) —
> not CS-Flow/AST-reported numbers, but M3DM's own — using **full XYZ geometry
> (not depth-only)**, while keeping CPU efficiency where feasible.

## Rules

1. **Never fabricate.** Any number in any report/table is either measured by us
   or explicitly cited as reported by prior work.
2. Never tune on test anomalies or test-AUROC to pick image-score aggregation.
3. Never silently change resolution between training and evaluation.
4. Never equate AUPRO definitions across papers; always state the integration
   limit (MVTec/AUPRO@0.3 here).
5. Always attribute M3DM, G2SF, etc. Properly cite.
6. **Reported-vs-measured table for every prior work we compare against.**
7. If reproduction of official numbers fails, **STOP and diagnose** before
   creating new methods.
8. Freeze the original XMV-AD Lite track first (done in Phase H0).
9. No git init / no git usage required for this track; isolated dirs only.
10. Never degrade dataset or labels (e.g., no 800x800→224 with lossy resize and
    silent re-scoring), and never silently change test protocol.
11. GPU is required for official M3DM reproduction; CPU-only machine recorded as
    a limitation, not an excuse to silently alter the method.
12. Paper-reported metrics are taken from the **primary source** (arXiv page /
    paper), never from a secondary table.

## Deliverables map (documents)

| Doc | Content |
|---|---|
| `docs/HIGH_ACCURACY_TRACK_START.md` | Freeze declaration, no-git decision, hardware ruling, isolated namespace |
| `docs/H0_M3DM_REPRODUCTION.md` | **H0 report (A–O)** — M3DM reproduction attempt & verdict |
| `experiments/high_accuracy/tables/H0_m3dm_reproduction.csv` | Reported-vs-measured table (all measured = NA: no GPU) |
| `experiments/high_accuracy/tables/H0_m3dm_paper_metrics.md` | M3DM paper numbers from primary source |
| `experiments/high_accuracy/runs/cpu_microbenchmark_*.json` | CPU runtime estimate (not a reproduction) |
| `experiments/high_accuracy/environment.json` | Hardware/software audit |
| `experiments/high_accuracy/dataset_inventory.json` | Dataset layout/counts |
| `experiments/high_accuracy/frozen_artifact_manifest.csv` | SHA256 of all frozen XMV-AD Lite artifacts |

## Phase status

- **H0 (authorized, done pending review):** environment audit, freeze/hash,
  M3DM pin, dataset/preprocess verification, paper numbers, CPU estimate, A–O
  report. **Verdict: FIX REPRODUCTION FIRST** (CUDA GPU required; this machine
  cannot run official M3DM).
- **H0B / H1 / later:** NOT STARTED. Out of scope until H0 report is reviewed
  and a CUDA-capable node is available.

## Directory layout (isolated)

```
src/xmvad/high_accuracy/          H-track package (empty until H1)
configs/high_accuracy/            H-track configs
scripts/high_accuracy/            H-track tooling
experiments/high_accuracy/        runs/ features/ memory_banks/ predictions/ tables/ figures/
third_party/m3dm/                 official M3DM repo (pinned bd8151a45..., read-only)
envs/high_accuracy/               isolated venv (timm 1.0.29, torch cpu) for benchmarking
```