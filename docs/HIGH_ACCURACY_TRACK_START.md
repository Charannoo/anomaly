# XMV-AD-H: High-Accuracy Research Track — Start Record

> Authoritative record: this directory `docs/high_accuracy/` + the track-level
> `docs/HIGH_ACCURACY_TRACK.md` hold the master plan for the accuracy-first
> research track **XMV-AD-H**. This file records the freeze point of the original
> XMV-AD Lite track and the isolated working rules for XMV-AD-H.

## 1. Freeze declaration

The original **XMV-AD Lite** research track (E1–E7 checkpoint evolution, Phase 1–6
metric tables, CPU deployment artifacts, attribution artifacts, metric
implementations, and the E4 config object used for deployment) is **COMPLETE** and
**FROZEN** as of 2026-09-18 (Phase 6 deployment completion, report
`docs/PHASE6_DEPLOYMENT.md`).

**Original XMV-AD Lite research track frozen before XMV-AD-H development.
No checkpoint, metric table, config, doc, metric-implementation file, or
deployment artifact of the frozen track will be modified, retrained, or
re-exported to chase XMV-AD-H goals.**

## 2. Repository state and integrity management

- `xmv-ad` is **not a git repository** (`git rev-parse` fails). Per roadmap
  decision, we do **NOT** initialize or mutate git history for this task.
- Integrity is instead protected by a **frozen-artifact hash manifest**:
  - `experiments/high_accuracy/frozen_artifact_manifest.csv` (per-file full
    SHA256 + bytes for 341 files, 151,393,314 bytes total)
  - `experiments/high_accuracy/frozen_artifact_manifest.json` (summary)
  - Generator: `scripts/high_accuracy/hash_frozen_artifacts.py`
- All XMV-AD-H work is isolated in dedicated directories (section 4) so the
  frozen tree can be re-verified after any phase without ambiguity.

## 3. Hardware-limitation decision (roadmap-required)

Audit (`experiments/high_accuracy/environment.json`, 2026-09-18T16:51:21+00:00):

- Windows 11, AMD64; host `Hiran`; 4 physical / 8 logical cores; **8.42 GB RAM
  total (0.91 GB available at audit time)**.
- **No CUDA GPU** (`torch.cuda.is_available() == False`, no GPUs).
- Python 3.13.6, torch 2.14.0+cpu; openvino 2026.4.0; onnxruntime 1.30.0; scipy;
  sklearn; matplotlib available.

Official M3DM requires CUDA-compiled `pointnet2_ops` and `knn_cuda` (see
`docs/H0_M3DM_REPRODUCTION.md` §C/M). On this CPU-only machine the official code
cannot import or run these ops, and a CPU fallback re-implementation is explicitly
**not** an M3DM reproduction. Per roadmap rule: we do NOT silently attempt a
multi-day CPU run; we prepare the pipeline, benchmark a small subset, report the
runtime estimate, and record the limitation. The method is not changed because of
hardware.

## 4. XMV-AD-H isolated namespace

```
scripts/    high_accuracy/      # H-track tooling (audit, hash, benchmark, ...)
configs/    high_accuracy/      # H-track experiment configs
src/        xmvad/high_accuracy/  # H-track Python package (isolated)
experiments/high_accuracy/      # H-track outputs
    runs/ features/ memory_banks/ predictions/ tables/ figures/
docs/       H0_M3DM_REPRODUCTION.md, HIGH_ACCURACY_TRACK.md, high_accuracy/
third_party/ m3dm/              # official M3DM repo (pinned, read-only)
```

Official M3DM source is pinned at commit
`bd8151a4563ee7445bdc32d01b2a3b384e4d54f5`
("The initial code", 2023-04-10, upstream `AnomalyDetection-M3DM` =
paper GitHub `nomewang/M3DM`).

## 5. H0 authority

For Phase **H0 only** the authorized work is: environment audit, freeze/hash,
M3DM repo pin + config/code understanding, dataset/preprocessing verification,
isolated-env plan, extraction of paper-reported M3DM numbers from the primary
source, a CPU-microbenchmark runtime estimate, and the honest A–O reproduction
report `docs/H0_M3DM_REPRODUCTION.md`. **H0B / H1 and any new method work are
out of scope until H0 is reviewed.**