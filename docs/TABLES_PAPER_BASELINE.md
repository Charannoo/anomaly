# Paper baseline table (Phase 5)

All values carry a provenance label. '--' means the value is not
reported in the cited source. AUPRO uses FPR limit 0.3 everywhere;
RESOLUTION notes at which map resolution the pixel metrics were
evaluated (official full-resolution vs 256).

| Method | Input | I-AUROC | P-AUROC | AUPRO | Params | Size (KB) | CPU lat. (ms) | Modality explanation | Res | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| Voxel f-AnoGAN | 3D only | 0.538 | -- | 0.583 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| Voxel f-AnoGAN | 3D+RGB | 0.517 | -- | 0.639 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| Voxel AE | 3D+RGB | 0.538 | -- | 0.564 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| Voxel Variation Model | 3D only | 0.699 | -- | 0.492 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| Voxel Variation Model | 3D+RGB | 0.609 | -- | 0.471 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| Depth Variation Model | 3D+RGB | 0.555 | -- | 0.335 | -- | -- | -- | Not applicable | native | Reported by prior paper |
| M3DM (Wang et al., CVPR 2023) | RGB+3D point cloud | 0.945 | -- | 0.964 | -- | -- | -- | N/A (pre-trained ViT + point-cloud memory banks) | native | Reported by prior paper |
| E1 RGB (Phase 2 student) | RGB only | 0.6401 | 0.895 | 0.6855 | 959097 | 4007.1 | 14.8 | No (single modality) | 256 | Measured by us |
| E2 Depth (Phase 2 student) | Z-depth only | 0.534 | 0.7542 | 0.4918 | 959097 | 4005.1 | 15.0 | No (single modality) | 256 | Measured by us |
| E3A Fixed fusion (ex-pred score mean) | RGB+Z-depth (late) | 0.64 | 0.9148 | 0.7224 | 0 | 0.0 | 30.3 | Partial (heatmap only) | 256 | Measured by us |
| E4 XMV-AD Residual Fusion | RGB+Z-depth (feature) | 0.6536 | 0.9582 | 0.8572 | 8738 | 42.3 | 34.0 | Yes (A_RGB / A_DEPTH / A_FINAL + relative evidence) | native-GT | Measured by us |
| E7 XMV-Attention Ablation | RGB+Z-depth (feature) | 0.6337 | 0.9179 | 0.7312 | 12770 | 60.2 | 38.0 | Yes (evidence) | ablated cross-attention | 256 | Measured by us |