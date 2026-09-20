# Gamma interventions on frozen E4 fusion (Phase 5)

Gate zeroed tests (no retraining). Delta = score(full) - score(gate);
a positive Delta means that residual gate contributes to the decision.
Scores are image-max of the calibrated fused mean map (valid-depth
masking). Evidence is relative anomaly evidence, not a causal claim.

## Pooled by injected kind x severity (mean Delta)

| kind       | severity   |   n |   s_full |   delta_rgb_resid |   delta_depth_resid |   delta_no_resid |
|:-----------|:-----------|----:|---------:|------------------:|--------------------:|-----------------:|
| depth_only | medium     |  20 |   6.8947 |           -0.5757 |             -0.0528 |          -0.6348 |
| depth_only | mild       |  20 |   6.7858 |           -0.6846 |             -0.0343 |          -0.7162 |
| depth_only | strong     |  20 |   6.9184 |           -0.5755 |             -0.0888 |          -0.6188 |
| joint      | medium     |  20 |  15.213  |           -0.5766 |             -0.0259 |          -0.5858 |
| joint      | mild       |  20 |  11.0186 |           -1.267  |             -0.005  |          -1.272  |
| joint      | strong     |  20 |  15.5635 |            0.1259 |              0.0082 |           0.1292 |
| rgb_only   | medium     |  20 |  15.1994 |           -0.9557 |              0      |          -0.9557 |
| rgb_only   | mild       |  20 |   9.8185 |           -1.3598 |              0      |          -1.3598 |
| rgb_only   | strong     |  20 |  15.5191 |           -0.5223 |              0      |          -0.5223 |

## Per-category gates and Delta summary

| category   |   gamma_rgb |   gamma_depth |   n_samples |
|:-----------|------------:|--------------:|------------:|
| bagel      |   -0.283283 |     -0.306303 |         180 |

## Per-category x kind x severity

| category   | kind       | severity   |   n |   s_full |   delta_rgb_resid |   delta_depth_resid |   delta_no_resid |
|:-----------|:-----------|:-----------|----:|---------:|------------------:|--------------------:|-----------------:|
| bagel      | depth_only | medium     |  20 |   6.8947 |           -0.5757 |             -0.0528 |          -0.6348 |
| bagel      | depth_only | mild       |  20 |   6.7858 |           -0.6846 |             -0.0343 |          -0.7162 |
| bagel      | depth_only | strong     |  20 |   6.9184 |           -0.5755 |             -0.0888 |          -0.6188 |
| bagel      | joint      | medium     |  20 |  15.213  |           -0.5766 |             -0.0259 |          -0.5858 |
| bagel      | joint      | mild       |  20 |  11.0186 |           -1.267  |             -0.005  |          -1.272  |
| bagel      | joint      | strong     |  20 |  15.5635 |            0.1259 |              0.0082 |           0.1292 |
| bagel      | rgb_only   | medium     |  20 |  15.1994 |           -0.9557 |              0      |          -0.9557 |
| bagel      | rgb_only   | mild       |  20 |   9.8185 |           -1.3598 |              0      |          -1.3598 |
| bagel      | rgb_only   | strong     |  20 |  15.5191 |           -0.5223 |              0      |          -0.5223 |

Note: per-sample Delta uses the image-max of the calibrated fused map;
samples with full score <= 0.5 (no anomaly signal) are dropped from the
mean to keep signal-to-noise interpretable.
gate legend: full / gamma_rgb=0 / gamma_depth=0 / both=0.