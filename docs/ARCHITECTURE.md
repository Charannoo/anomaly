# Architecture (XMV-AD)

## Design philosophy

M3DM: heavy pretrained RGB + point-cloud reps -> alignment -> fusion ->
multiple memory banks -> decision.

XMV-AD (final identity, E4 "Residual Fusion"): frozen RGB + Z-depth students
-> tiny zero-init residual concat @1/16 with per-modality gamma gates ->
modality-preserving heads -> A_RGB + A_DEPTH -> calibrated (Phase 3A) fusion ->
A_FINAL -> validated attribution.

Historical note: E5/E6/E7 answer RQ2 with a bidirectional cross-attention
exchanger and are reported as the published ablation "XMV-Attention Ablation".
They do **not** beat E4 (see `docs/PHASE4B_ATTENTION_ABLATION.md`). The
cross-attention path is documented below for provenance, but is NOT the
released model.

## Encoders (Phase 2, frozen in E4)

- MobileNetV3-Small RGB + separate MobileNetV3-Small depth (1ch input, no
  weight sharing; depth stem initialized by averaging RGB kernels).
- Features at ~1/8 and ~1/16; teacher + student distillation head at both
  scales; anomaly map = cosine discrepancy between frozen teacher and student
  features (per-scale weighted 0.5/0.5).

## E4 residual feature concatenation (Phase 4A, final fusion)

- Frozen Phase 2 student/teacher features; only ConcatFusion is trainable
  (8,738 params).
- S_rgb16/S_depth16 (48 ch) -> 1x1 proj to d=32 -> concat -> light block ->
  heads -> per-modality residuals added through zero-init scalar gates:
  `S_rgb_fused = S_rgb + gamma_rgb * delta_rgb`, same for depth.
- gamma init 0 => E4 starts as the Phase 2 identity; fusion only redistributes
  evidence if the residual is earned.
- Full protocol in `docs/PHASE4A_CONCAT_FUSION.md`.

## Cross-attention ablation (Phase 4B, NOT the final model)

Q_rgb=Wq(P_rgb), K/V from depth and vice versa; scaled dot-product, 1-2 heads,
learnable residual scales + LayerNorm, decoder @1/16. Attention matrices were
stored only as *interaction visualizations*, never as causal proof. Ablated
out: pooled image-level -0.040 vs E4 (AUPRO .73 vs .85).

## Modality preservation (non-negotiable)

A_RGB and A_DEPTH are produced BEFORE A_FINAL. Final fusion = calibrated 50/50
mean with valid-depth masking (invalid depth -> RGB only), plus image score =
max of the fused calibrated map. Attribution is *relative evidence*, never a
causal claim.

## Anomaly learning (Phase 2)

Normal-only distillation: frozen teacher vs trainable student, cosine
discrepancy per level, L = L_rgb + lambda_depth*L_depth.

## Status

Phase 1 (data pipeline), Phase 2 (students), Phase 3 (calibration/late
fusion), Phase 4A (E4 residual fusion), Phase 4B (cross-attention ablation),
Phase 5 (native-GT official metrics + modality-aware explainability) complete.
No INT8/ONNX/OpenVINO/dashboard.