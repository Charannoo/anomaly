# PHASE3B_FROZEN — E3B reliability-aware score fusion

Phase 3B is APPROVED and FROZEN. Later phases may read its tables but must
not silently alter its rules, statistics, or results.

- Freeze date: 2026-09-18. No VCS; identity = this doc + calibration JSONs +
  metric tables + Phase 2 checkpoint hashes (re-verified identical).
- Artifacts: `experiments/calibration/E3B_reliability_stats.json`,
  `experiments/tables/E3B_{reliability_metrics,vs_all,oracle,defect_level,error_correlation}.csv`.
- Code: `src/xmvad/fusion/score_fusion.py` (3 pre-registered rules),
  `src/xmvad/analysis/oracle.py` (TEST-ORACLE, analysis only),
  `scripts/compute_reliability_stats.py`, `scripts/run_reliability_fusion.py`.
- No retraining; no test data in any parameter; evidence T=1 pre-registered.

## Three tested rules (MEAN results)

| Method | I-AUROC max | I-AUROC top1% | P-AUROC | AUPRO |
|---|---|---|---|---|
| RGB / Depth / FixedMean | .640/.534/.640 | .686/.588/.732 | .895/.754/.915 | .685/.492/.722 |
| E3B-1 Static (normal-noise) | .659 (+.019 bs) | .752 (+.066) | .888 (−.007) | .660 (−.025) |
| E3B-2 Validity-adjusted | .662 (+.022) | .752 (+.065) | .894 (−.001) | .674 (−.012) |
| E3B-3 Evidence-adaptive T=1 | .596 (−.045) | .665 (−.021) | .893 (−.002) | .671 (−.014) |

(bs = vs best-single.) Static/validity help image-level, surrender
localization. Evidence-max chases noise (carrot evidence I = depth exactly).

## Negative transfer (fixed → static/validity/evidence, max I-AUROC)

peach .422→.453/.442/.519 (all fix); potato .463→.613/.607/.604 (fixed, but
potato AUPRO .820→.592: detection fixed by sacrificing localization);
rope .690→.763/.763/.662; tire .559→.587/.586/.500; cable_gland .802→
.749/.804/.626; carrot .696→unchanged/.546; foam .744→unchanged/.611.

## Oracle diagnostic (TEST-ORACLE, analysis only)

Oracle I-AUROC MEAN .941 vs best-single .640 (+.30); per-category .85–1.0.
Perfect modality selection would transform detection — large headroom,
motivating supervised reliability gating (Phase 3C).

## Error correlations

RGB vs depth max-score Pearson/Spearman ≈ 0 on normals and anomalous in all
categories except rope-anomalous (+.57/+.49). Branches fail on different
samples: complementarity is real.

## Why one reliability definition fails both tasks

Normal-noise reliability predicts *detection* weighting usefully but is a
poor proxy for *localization* reliability (potato: image +0.15 while AUPRO
−0.23 under static weights). Detection and localization reliability must be
modeled as separate heads — the Phase 3C dual-gate design follows directly.
