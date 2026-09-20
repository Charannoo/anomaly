# Controlled modality-attribution metrics (Phase 5)

Relative anomaly evidence on CALIBRATED branch maps; region = injected
production-like synthetic mask. Not a causal claim. Data: TRAIN-NORMAL only.

Categories: bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire;  severities: medium, mild, strong;  total synthetic samples: 3600

## Per-category (pooled over severity)

| category    |   dom_acc |   macro_f1 |   consistency |   p_rgb_ratio |   pa_rgb |   pa_depth |   pa_fused |   n |
|:------------|----------:|-----------:|--------------:|--------------:|---------:|-----------:|-----------:|----:|
| bagel       |    0.7806 |     0.8647 |        0.8056 |        1.8047 |   0.8368 |     0.8451 |     0.9356 | 360 |
| cable_gland |    0.8139 |     0.8863 |        0.75   |        1.3675 |   0.8539 |     0.831  |     0.9393 | 360 |
| carrot      |    0.9417 |     0.9688 |        0.9556 |        1.2002 |   0.8464 |     0.8845 |     0.9696 | 360 |
| cookie      |    0.6972 |     0.8048 |        0.8917 |        2.0112 |   0.847  |     0.8686 |     0.9612 | 360 |
| dowel       |    0.9361 |     0.9659 |        0.9611 |        1.5088 |   0.8875 |     0.901  |     0.9869 | 360 |
| foam        |    0.8278 |     0.8947 |        0.925  |        1.1206 |   0.8324 |     0.8095 |     0.9417 | 360 |
| peach       |    0.6167 |     0.7323 |        0.7333 |        3.6109 |   0.8709 |     0.8096 |     0.9472 | 360 |
| potato      |    0.8611 |     0.9146 |        0.7556 |        2.6563 |   0.8479 |     0.82   |     0.9576 | 360 |
| rope        |    0.525  |     0.5831 |        0.7417 |        0.3219 |   0.8548 |     0.9198 |     0.9931 | 360 |
| tire        |    0.8139 |     0.8885 |        0.9556 |        1.4239 |   0.8658 |     0.8554 |     0.9538 | 360 |

## Per-severity (pooled over categories)

| severity   |    n |   dom_acc |   macro_f1 |   consistency |   mean_p_rgb |   evid_ratio_rgb_over_depth |   acc_rgb_only |   p_rgb_rgb_only |   acc_depth_only |   p_rgb_depth_only |   acc_joint |   p_rgb_joint |
|:-----------|-----:|----------:|-----------:|--------------:|-------------:|----------------------------:|---------------:|-----------------:|-----------------:|-------------------:|------------:|--------------:|
| medium     | 1200 |    0.8125 |     0.8931 |        0.8367 |       0.5549 |                      1.4562 |         0.9525 |           0.863  |           0.7125 |             0.2343 |      0.7725 |        0.5675 |
| mild       | 1200 |    0.7125 |     0.818  |        0.8125 |       0.5776 |                      1.5906 |         0.92   |           0.8515 |           0.4725 |             0.3208 |      0.745  |        0.5607 |
| strong     | 1200 |    0.8192 |     0.8992 |        0.8933 |       0.5449 |                      1.3461 |         0.9125 |           0.8487 |           0.7825 |             0.2035 |      0.7625 |        0.5824 |
| all        | 3600 |    0.7814 |     0.8729 |        0.8475 |       0.5591 |                      1.4609 |         0.9283 |           0.8544 |           0.6558 |             0.2528 |      0.76   |        0.5702 |

## Confusion blocks (predicted class column x injected modality row, 
fraction of samples) on pooled data

| injected \ predicted | RGB-DOMINANT | DEPTH-DOMINANT | JOINT |
|---|---|---|---|
| rgb_only | 0.928 | 0.003 | 0.068 |
| depth_only | 0.022 | 0.656 | 0.322 |
| joint | 0.138 | 0.099 | 0.760 |

## Notes
- AttributionConsistency: (Delta_RGB vs Delta_DEPTH) agrees with 
  (p_rgb vs 0.5) - decision-ablation direction aligns with evidence direction.
- Spatial pixel-AUROC: branch/ fused anomaly map against the injected mask.
- UNCERTAIN class (decisions with effectively no positive evidence) is a 
  correctness neutral outcome; use apply to cases with decisive evidence.
- macro-F1: correct prediction -> injected kind, otherwise 'MISS'
  (wrong dominant class or UNCERTAIN); 3-class macro average. Corrected in
  the Phase 6 audit (previously joint was always scored correct).
