# Deployment input contract (Phase 6)

Status: contract for the frozen **E4 "XMV-AD Residual Fusion"** category models
(MVTec 3D-AD). This document fixes the runtime interface the exported
artifacts (`deployment/onnx/*.onnx`, `deployment/openvino_fp32/*.xml`,
`deployment/openvino_int8/*.xml`) must honour. It is a contract, not a
design tutorial: batch 1 is the primary deployment target; a fixed 256x256
input the only guaranteed spatial size.

## 1. Scope and workload

| Item | Value |
|---|---|
| Model family | E4 `E4ConcatModel` (RGB TeacherStudentAD + Depth TeacherStudentAD, frozen Phase 2, plus `ConcatFusion`, 8,738 trainable params) |
| Category models | 10 separate checkpoints (`checkpoints/E4_concat_fusion/<cat>_best.pt`), validated by `experiments/artifacts/e4_checkpoint_hashes.csv` |
| Input resolution | 256 x 256 (fixed). Batch 1 primary; batch >1 must not be assumed |
| Backends | PyTorch (reference), ONNX Runtime (CPU), OpenVINO FP32, OpenVINO INT8 |
| NEURAL graph | branch encoders + fusion + per-modality anomaly maps (see `A_rgb`/`A_depth`) |
| POST-PROCESSING | calibration + 50/50 fusion + image score (NOT part of the exported graph; see section 5) |

## 2. Input tensors (neural graph)

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `rgb` | `[1,3,256,256]` | float32 | ImageNet-normalized RGB: `(img/255 - mean)/std`, mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` (CHW tensor from channel-last PIL, channel order RGB) |
| `depth` | `[1,1,256,256]` | float32 | normalized depth (robust-mean inverted depth / scale, clipped `[-3,3]`); invalid pixels zeroed |
| `valid` | `[1,1,256,256]` | float32 or bool | 1.0 where a depth sample exists, else 0.0 (NEAREST-resized from raw validity) |

Precondition (documented, enforced by callers): all three are dense CHW
tensors in **[0,1) float32 RGB -> normalized by the SAME per-category
parameters used in evaluation**; `valid` is exactly the nearest-neighbour
downsample of the sensor mask. The neural graph never assumes these
normalization constants internally (they are applied before entry).

## 3. Output tensors (neural graph)

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `A_rgb` | `[1,1,256,256]` | float32 | RGB-branch anomaly map (cosine discrepancy at stage8+b16, gaussian-smoothed, interpolated to 256x256) |
| `A_depth` | `[1,1,256,256]` | float32 | depth-branch anomaly map (same protocol; depth-invalid pixels zeroed) |

`A_rgb` / `A_depth` are RAW anomaly maps (higher = more anomalous). They are
NOT calibrated and NOT reliability-fused; downstream post-processing applies
the frozen per-category calibration + fusion.

## 4. Domain and conditioning invariants

- Anomaly maps are non-negative after the divergence clamp (reLU clamp on
  cosine distance). They may contain the fusion residual for the depth branch
  which is masked at depth-invalid pixels (zero there).
- Values are finite everywhere (small machine-epsilon jitter only). No
  NaN/inf expected; validity masks keep depth out of the divergence loss.

## 5. POST-PROCESSING (frozen, deterministic, EXCLUDED from the exported graphs)

Kept outside the neural graph so quantization statistics are computed only on
the neural part. Same operations as the frozen evaluation protocol:

1. **Calibration**: `z_rgb = (A_rgb - median_rgb) / (scale_rgb + eps)`,
   `z_depth = (A_depth - median_depth) / (scale_depth + eps)`, per-category
   params from `experiments/tables/E4_concat_calibration.json` (validation-normal
   robust median/MAD fit, E3-declared).
2. **Reliability fusion (E4 default 50/50 mean)**: pixel-wise
   `A_final = where(valid, 0.5*z_rgb + 0.5*z_depth, z_rgb)`; the max secondary
   `fused_max = where(valid, max(z_rgb, z_depth), z_rgb)` is also reported.
3. **Image score**: `score = max(A_final)` (primary; the official MVTec
   protocol). Secondary variants `topk_0.1pct` / `topk_1pct` mean-of-top-k.

This is the exact post-processing in `scripts/evaluate_concat.py` + 
`src/xmvad/fusion/score_fusion.py` + `src/xmvad/metrics/scores.py`.

## 6. Latency / memory definition (contractual)

- **model-only latency**: time of the exported neural graph only (runtime
  `run`/`infer` for a batch-1 256 input), after warmup 50 runs; report
  p50/p90/p95/p99 over 500 timed runs, single-threaded unless documented.
- **end-to-end latency**: model-only + post-processing (calibration, fusion,
  image score) on the same hardware.
- Peak RSS measured at process level for model-only and end-to-end runs.
- Same-machine only in the Pareto plot; paper numbers are never mixed.

## 7. Accuracy-Loss acceptance thresholds (INT8 pre-declared)

| Loss | Threshold | Meaning |
|---|---|---|
| mean \|INT8-FP32\| metric delta (I-AUROC, P-AUROC, AUPRO) | `<= 0.005` | ideal |
| | `<= 0.01` | very acceptable |
| | `<= 0.02` | potentially acceptable, must disclose failing categories and cite the efficiency gain |
| per-category worst drop | `<= 0.01` | no single-category regression beyond FP32 uncertainty |

INT8 loss is reported per metric per category; every category must be listed.
If the mean loss exceeds 0.02 or a category is unusable, the report marks INT8
as failed for that category and the FP32 backend stays recommended.

## 8. Version-locked toolchain (reproduction)

- torch 2.14.0+cpu, onnx 1.22.0, onnxruntime 1.30.0, openvino 2026.4.0,
  nncf 3.4.0, python 3.13.6, numpy 2.4.6 (see `experiments/artifacts/environment.json`).
- Exports: opset 17 minimum; `onnx.checker` + shape-inference run on every
  exported `.onnx`; saved to `deployment/onnx/<cat>.onnx`.

## 9. Failure reporting (contractual)

If any export/quantization/validation step fails, the report records the step,
the backend, the category, and the error. We do not silently substitute a
different architecture or re-tune on test anomalies. FP32 remains the default
recommended backend unless INT8 passes section 7.