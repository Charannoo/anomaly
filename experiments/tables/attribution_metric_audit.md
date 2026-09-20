# Attribution-metrics reporting audit (Phase 6 milestone 1)

Recomputed from `experiments/attribution/attribution_data/<cat>.npz`.
This is a read-only audit: no thresholds or data subsets are changed.
It recomputes metric values against the frozen script that produced them.

## Claimed vs recomputed

- dominant-modality accuracy reported: **0.7814**;
  recomputed: **0.781389**
- macro-F1 originally reported: **0.9183** (from the frozen Phase-5
  script `eval_attribution.py`); faithful recomputed 3-class macro-F1
  (UNCERTAIN predicted = MISS): **0.872873**
- Discrepancy: the frozen script's `macro_f1()`/summary branch labelled
  every JOINT sample `'joint'` regardless of whether the dominant
  prediction was correct, so the JOINT label always scored F1=1.0
  (precision 1.0, recall 1.0), inflating the average to 0.9183.
  After correction the metric is computed as: correct prediction ->
  injected kind, otherwise 'MISS'. This is a metric-computation fix
  only; no thresholds, model outputs, or test-anomaly handling changed.

## Evaluated classes and sample sets

- Injected modality kinds: depth_only, joint, rgb_only
- No CLEAN / normal-only synthetic samples are included; all 3,600
  synthetic images are defect-injected.
- Predicted classes from attribution evidence: DEPTH-DOMINANT, JOINT, RGB-DOMINANT, UNCERTAIN.
- UNCERTAIN = region carries no decisive positive evidence; counted as
  incorrect (MISS) in dominant accuracy and macro-F1.

## Confusion matrix (true row x predicted column, pooled fraction)

|            |   rgb_only |   depth_only |   joint |
|:-----------|-----------:|-------------:|--------:|
| rgb_only   |          1 |            0 |       0 |
| depth_only |          0 |            1 |       0 |
| joint      |          0 |            0 |       1 |

## UNCERTAIN breakdown per injected kind

- `rgb_only`: 0.000%
- `depth_only`: 0.000%
- `joint`: 0.250%

## Per-severity accuracy (correct dominant-class count / samples)

- medium: 0.8125
- mild: 0.7125
- strong: 0.8192

## Notes

- macro-F1 uses `sklearn.metrics.f1_score(average='macro')` with
  `zero_division=0.0` over the 3 injected kinds; a UNCERTAIN or
  wrong-dominant prediction is labelled 'MISS' (incorrect).
- An alternative macro-F1 over {rgb_only,depth_only,joint,MISS} (4-class)
  weights the miss class differently; the reported number uses the 3-class
  variant for direct comparability with per-kind accuracy.
- All samples are treated identically across severity levels; severity is
  only used for stratified breakdowns.
