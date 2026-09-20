# Deployment (CPU)

## Chain

PyTorch -> ONNX -> ONNX Runtime CPU -> OpenVINO CPU. Training may use CUDA;
inference MUST support device=cpu with no CUDA requirement.

## Protocol (Phase 7)

`python scripts/benchmark.py --checkpoint checkpoints/xmvad.pt --device cpu`:
batch 1, warmup 50, >=200 iters, sizes 256/320/384, controlled threads.
Report p50/p95/mean ms, FPS, peak RAM, params, MB.

## Quantization (Phase 7)

FP32 first, then INT8 post-training quantization; report accuracy AND
latency/memory deltas. Investigate QAT if PTQ degrades detection. Never claim
INT8 superiority without measurement.

## Targets (goals, not results)

<10M params, <30MB optimized, CPU p95 <100ms (stretch <50ms), >=10 FPS,
<1.5GB peak RAM. Accuracy goal is a materially better
accuracy-efficiency-explainability trade-off, not a forced win over M3DM.
