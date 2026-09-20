# Baselines (M3DM comparison)

## Primary baseline

Wang, Yue et al. "Multimodal Industrial Anomaly Detection via Hybrid Fusion"
(M3DM / Multi-3D-Memory), CVPR 2023.

Components: RGB + 3D point-cloud features, Point Feature Alignment,
Unsupervised Feature Fusion, patch-wise contrastive learning, multiple memory
banks, decision-layer fusion.

## Fairness disclosure (mandatory in any comparison)

- M3DM consumes RGB + full XYZ point-cloud features.
- XMV-AD v1 consumes RGB + aligned Z-depth map (simpler, cheaper to deploy).
- Never present them as identical inputs. The representation gap is itself a
  research finding; a later XMV-AD-XYZ (X,Y,Z 3ch input) variant will narrow it:
  XMV-RGBD vs XMV-XYZ vs M3DM.

## Shared metrics

Image-AUROC, Pixel-AUROC, AUPRO under the MVTec 3D-AD protocol.

## Status

No M3DM numbers are reproduced or claimed in Phase 1. Reported-vs-reproduced
vs XMV-AD numbers will be labeled explicitly once experiments exist.
