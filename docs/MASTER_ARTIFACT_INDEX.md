# Master Artifact Index: Repository Artifacts & Research Provenance

This index catalogs all code, configuration files, checkpoints, feature caches, coresets, raw prediction arrays, verification scripts, audits, and documentation artifacts across the repository.

---

## 1. Release & Verification Snapshot (`release/H5D_PNTC_VERIFIED/`)

This directory contains the immutable research snapshot for the final frozen model (Git tag: `h5d-pntc-verified`, Commit: `4d1f448`).

| Artifact Relative Path | Purpose / Description | Canonical Status | Phase Provenance | Key Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| `release/H5D_PNTC_VERIFIED/H5D_CANONICAL_CONFIG.yaml` | Master frozen hyperparameters ($\lambda=0.35, k=5, M=15K, \sigma=3.8$) | `CANONICAL` | Phase H5 | PyYAML |
| `release/H5D_PNTC_VERIFIED/H5_reproducibility_hashes.json` | Complete SHA256 checksums of configs, raw predictions, and environment | `CANONICAL` | Verification | Python hashlib |
| `release/H5D_PNTC_VERIFIED/H5_raw_metric_verification.csv` | Recomputed unrounded AUROC & AUPRO metrics directly from `.npz` arrays | `CANONICAL` | Verification | `h5d_raw_predictions_replay.npz` |
| `release/H5D_PNTC_VERIFIED/H5_baseline_repro_check.csv` | Exact category-by-category baseline invariance check ($\Delta = 0.0$) | `CANONICAL` | Verification | H4B feature cache |
| `release/H5D_PNTC_VERIFIED/H5_seed_audit.csv` | Multi-seed stability audit (Seeds 42, 100, 2026) | `CANONICAL` | Verification | Coreset sampler |
| `release/H5D_PNTC_VERIFIED/H5_pair_integrity.json` | Programmatic audit confirming 0 paired-prototype ID misalignments | `CANONICAL` | Verification | Paired coreset arrays |
| `release/H5D_PNTC_VERIFIED/H5_topology_statistics.csv` | Normal vs anomaly empirical JS divergence distributions | `CANONICAL` | Analysis | Test prediction arrays |
| `release/H5D_PNTC_VERIFIED/H5_ablation_integrity.md` | Audit of progressive ablation trajectory (H4B $\to$ H5A $\to$ H5B $\to$ H5C $\to$ H5D) | `CANONICAL` | Phase H5 | Ablation CSVs |
| `release/H5D_PNTC_VERIFIED/H5_hyperparameter_provenance.md` | Formal audit proving zero test-label tuning or hyperparameter leakage | `CANONICAL` | Verification | Normal training statistics |
| `release/H5D_PNTC_VERIFIED/H4B_lineage_audit.md` | Reconciles exploratory Table A (~0.9562) vs canonical Table B (0.95401000) | `CANONICAL` | Phase H4/H5 | Baseline CSVs |
| `release/H5D_PNTC_VERIFIED/H5_PNTC_FORMULATION.md` | Mathematical specification of PNTC retrieval and JSD topology scoring | `CANONICAL` | Phase H5 | — |
| `release/H5D_PNTC_VERIFIED/H5_NOVELTY_AUDIT.md` | 2023–2026 literature comparison (M3DM, CFM, CPIR, Attn-Mem, G2SF) | `CANONICAL` | Phase H5 | Published literature |
| `release/H5D_PNTC_VERIFIED/PAPER_TABLES.md` | Camera-ready LaTeX and markdown comparison tables | `CANONICAL` | Publication | All verified results |
| `release/H5D_PNTC_VERIFIED/PNTC_STATISTICAL_REPORT_VERIFIED.md` | Exact $df=9$ paired $t$-test and Wilcoxon signed-rank test outputs | `CANONICAL` | Verification | SciPy stats |
| `release/H5D_PNTC_VERIFIED/PRIOR_WORK_METRIC_AUDIT.md` | Primary source verification of CFM (0.954/0.993/0.971) and prior work | `CANONICAL` | Verification | CVPR/ICCV papers |
| `release/H5D_PNTC_VERIFIED/PNTC_CLAIMS_DISCIPLINE.md` | Bounded claims checklist forbidding SOTA claims and mandating caveats | `CANONICAL` | Publication | Verification docs |

---

## 2. Experimental Results & Verification Artifacts (`results/`)

| Artifact Relative Path | Purpose / Description | Canonical Status | Phase Provenance |
| :--- | :--- | :--- | :--- |
| `results/H5D_CANONICAL_CONFIG.yaml` | Active canonical configuration for H5-D PNTC | `CANONICAL` | Phase H5 |
| `results/h5d_raw_predictions_replay.npz` | Immutable raw test predictions, maps, and masks for all 1,197 samples | `CANONICAL` | Verification |
| `results/H5_raw_metric_verification.csv` | Ground-truth numerical verification table matching canonical metrics | `CANONICAL` | Verification |
| `results/H5_per_category_pntc.csv` | Category-level metrics for final PNTC ($0.96541 / 0.99416 / 0.96939$) | `CANONICAL` | Phase H5 |
| `results/H5_pntc_ablations.csv` | Quantitative ablation data for H5-A, H5-B, H5-C, H5-D | `CANONICAL` | Phase H5 |
| `results/H5_baseline_repro_check.csv` | Zero-delta verification of H4B baseline | `CANONICAL` | Verification |
| `results/H5_reproducibility_hashes.json` | Hash records for data integrity | `CANONICAL` | Verification |
| `results/H5_seed_audit.csv` | Seed sensitivity evaluation outputs | `CANONICAL` | Verification |
| `results/H5_pair_integrity.json` | Paired-ID integrity audit results | `CANONICAL` | Verification |
| `results/H5_topology_statistics.csv` | Empirical divergence distribution data | `CANONICAL` | Analysis |
| `results/H4B_DINOv2_PointMAE.csv` | Category metrics for Canonical H4B ($0.95401000 / 0.99109000 / 0.95852000$) | `CANONICAL` | Phase H4 |
| `results/H4A_DINOv2_RGB.csv` | DINOv2 single RGB baseline results | `EXPLORATORY` | Phase H4 |
| `results/H4C_DINOv2_PointMAE_Concat.csv` | DINOv2 + PointMAE feature concatenation results | `EXPLORATORY` | Phase H4 |
| `results/H3D_decision_fusion.csv` | DINO + PointMAE decision fusion with `mean_top_0.5%` ($0.8979 / 0.9897$) | `EXPLORATORY` | Phase H3.5 |
| `results/H35_image_scoring.csv` | Image pooling comparison (max, top-k%, reweighted) | `EXPLORATORY` | Phase H3.5 |
| `results/H3C_corrected.csv` | Corrected feature concatenation after 2D spatial bug fix | `EXPLORATORY` | Phase H3 |
| `results/H3C_alignment_audit.md` | Technical post-mortem on legacy `(4*a+c)%784` spatial bug | `AUDIT` | Phase H3 |
| `results/pointmae_checkpoint_audit.md` | Detailed audit of 53 unexpected Point-MAE decoder keys | `AUDIT` | Phase H2 |

---

## 3. Core Source Code & Execution Scripts

| Script Relative Path | Function / Responsibility | Track | Dependencies |
| :--- | :--- | :--- | :--- |
| `verify_h5_metrics_raw.py` | Standalone metric evaluator computing AUROC/AUPRO from raw `.npz` arrays | Track 2 | `numpy`, `sklearn`, `scipy` |
| `experiments/high_accuracy/src/extract_features.py` | GPU feature extractor for DINOv2 and Point-MAE with disk caching | Track 2 | `torch`, `timm`, `open3d` |
| `experiments/high_accuracy/src/pntc_core.py` | Vectorized PNTC prototype retrieval, JSD calculation, and confidence gating | Track 2 | `numpy`, `scipy` |
| `experiments/high_accuracy/src/coreset.py` | MinMax greedy facility location coreset sampler for joint descriptors | Track 2 | `numpy` |
| `experiments/high_accuracy/src/evaluator.py` | Official 2D bilinear interpolation, Gaussian smoothing, and AUPRO engine | Track 2 | `numpy`, `scipy`, `cv2` |
| `src/models/xmv_residual_fusion.py` | Lightweight E4 model with MobileNetV3 teacher-student residual fusion | Track 1 | `torch`, `torchvision` |
| `src/deployment/export_onnx.py` | ONNX export and runtime verification pipeline for lightweight E4 | Track 1 | `onnx`, `onnxruntime` |

---

## 4. Documentation & Research Reference Files (`docs/`)

| Documentation File | Core Topic / Content |
| :--- | :--- |
| `docs/MASTER_PROJECT_HISTORY.md` | Exhaustive chronological narrative from MobileNetV3 to PNTC publication freeze |
| `docs/MASTER_EXPERIMENT_TIMELINE.md` | Tabular registry of every hypothesis, architecture, parameter count, metric, and outcome |
| `docs/MASTER_ARCHITECTURE_AND_DECISIONS.md` | Technical justifications for all components, data flow diagram, and failure analysis |
| `docs/MASTER_RESULTS_REGISTRY.md` | Authoritative metric registry classifying canonical, ablation, exploratory, and prior work |
| `docs/MASTER_ARTIFACT_INDEX.md` | Complete artifact catalog and file dependency index (this document) |
| `docs/H5_PNTC_FORMULATION.md` | Mathematical derivation of paired memory, soft distributions, and confidence gating |
| `docs/H5_NOVELTY_AUDIT.md` | Comprehensive literature review across 2023–2026 multimodal RGB-3D papers |
| `docs/PNTC_METHOD_PAPER.md` | Formal paper methodology draft with academic notation |
| `docs/PNTC_METHOD_ARCHITECTURE.md` | Structural breakdown of the PNTC inference engine |
| `docs/PNTC_STATISTICAL_REPORT_VERIFIED.md` | Statistical hypothesis testing report ($df=9$ paired $t$-test and Wilcoxon tests) |
| `docs/PNTC_FAILURE_ANALYSIS.md` | Comprehensive catalog of negative results and engineering lessons |
| `docs/PNTC_CLAIMS_DISCIPLINE.md` | Scientific boundaries and mandatory citation/protocol disclosures |
| `docs/PNTC_TOPOLOGY_EXPLAINER.md` | Intuitive explanation of retrieval topology vs feature distance for viva/Q&A |
| `docs/PNTC_QUALITATIVE_ANALYSIS.md` | Category-specific defect localization analysis (cookie, foam, potato, etc.) |
| `docs/PNTC_COMPLEXITY.md` | Latency, memory RSS, and parametric complexity analysis |
| `docs/PRIOR_WORK_METRIC_AUDIT.md` | Primary source audit of published metrics from M3DM, CFM, CPIR, Attn-Mem, G2SF |
| `docs/PAPER_TABLES.md` | Full LaTeX tables ready for paper insertion |
| `docs/PHASE6_DEPLOYMENT.md` | Edge deployment benchmarks, ONNX runtime profiles, and INT8 failure analysis |
