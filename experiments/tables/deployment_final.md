| backend | model p50 (ms) | e2e p95 (ms) | model FPS | peak RSS (MB) | file (MB) | mean I-AUROC | mean P-AUROC | mean AUPRO | mean|Δ| I/P/AUPRO vs frozen |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | 29.42 | 36.49 | 34.0 | 381.3 | 7.71 | 0.6536 | 0.9562 | 0.853 | 0.0/0.0/0.0 |
| onnx | 8.36 | 12.88 | 119.7 | 390.2 | 2.32 | 0.6536 | 0.9562 | 0.853 | 0.0/0.0/0.0 |
| openvino_fp32 | 11.52 | 16.32 | 86.8 | 411.7 | 1.02 | 0.6565 | 0.9548 | 0.8487 | 0.005/0.0014/0.0044 |
| openvino_int8 | 12.25 | 15.91 | 81.6 | 451.6 | 1.22 | 0.5707 | 0.8657 | 0.641 | 0.113/0.0905/0.212 |

INT8 verdict: INT8 exceeded worst-case mean-abs |delta| threshold (0.02): 0.212 (AUPRO mean abs delta 0.212). Declared failure path entered; FP32 stays recommended.