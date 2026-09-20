# BASE PAPER — MVTec 3D-AD (Bergmann et al., VISAPP 2022)

Formal reference for XMV-AD's dataset, protocol, and primary comparison point.
All numbers below are reproduced **exactly** from the published paper and are
labeled as **"Reported by prior paper"**. They were transcribed from the
author's open-access version (arXiv:2112.09045 / DOI 10.5220/0010865000003124).

> Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger. "The MVTec 3D-AD
> Dataset for Unsupervised 3D Anomaly Detection and Localization." Proceedings
> of the 17th International Joint Conference on Computer Vision, Imaging and
> Computer Graphics Theory and Applications — Volume 5: VISAPP, pp. 202–213,
> 2022. **Best Industrial Paper Award (VISAPP 2022).**

Source note: none of the numeric values in this document were reproduced from
memory; they were fetched from the paper's open-access text on 2026-09-18.

## 1. Dataset statistics (paper Table 1) — Reported by prior paper

| Category | #Train | #Val | #Test good | #Test anomalous | #Defect types | #Annotated regions | Image size (W×H) |
|---|---|---|---|---|---|---|---|
| bagel | 244 | 22 | 22 | 88 | 4 | 112 | 800×800 |
| cable gland | 223 | 23 | 21 | 87 | 4 | 90 | 400×400 |
| carrot | 286 | 29 | 27 | 132 | 5 | 159 | 800×800 |
| cookie | 210 | 22 | 28 | 103 | 4 | 128 | 500×500 |
| dowel | 288 | 34 | 26 | 104 | 4 | 131 | 400×400 |
| foam | 236 | 27 | 20 | 80 | 4 | 115 | 900×900 |
| peach | 361 | 42 | 26 | 106 | 5 | 131 | 600×600 |
| potato | 300 | 33 | 22 | 92 | 4 | 115 | 800×800 |
| rope | 298 | 33 | 32 | 69 | 3 | 72 | 900×400 |
| tire | 210 | 29 | 25 | 87 | 4 | 95 | 600×800 |
| **total** | **2656** | **294** | **249** | **948** | **41** | **1148** | — |

XMV-AD uses the identical 10 categories, splits, and GT annotations. Note the
non-uniform native resolutions (400×400 to 900×900), which motivates Phase 5's
full-resolution re-evaluation (see §4).

## 2. Acquisition and protocol facts — Reported by prior paper

- Sensor: Zivid One+ Medium (structured light), 1920×1200 raw scans, cropped to
  the fixed rectangular domains above; RGB + (x,y,z) per pixel.
- Training/validation = anomaly-free only; test = 249 good + 948 anomalous.
- GT: anomaly regions annotated on the 3D point cloud and mapped to 2D GT masks;
  anomalies that appear via 3D-reconstruction failure (invalid pixels) and via
  RGB-only appearance are included in GT.
- Official localization metric: per-region-overlap (PRO), averaged over GT
  connected components, integrated up to **FPR limit ≤ 0.3** and normalized.
  The paper explicitly recommends *not* integrating beyond FPR 0.3.
- Official detection metric: I-AUROC computed on the **maximum** anomaly score
  of each anomaly map.
- Official evaluation code is distributed separately (evaluation_code v2.0);
  our `src/xmvad/metrics/pro.py` implements its PRO/AUPRO path exactly for the
  curve code, with the documented difference that the official end-to-end
  script compares FULL-resolution maps (verification in Phase 5, §4).

## 3. Initial benchmark (paper Table 2 and Table 5) — Reported by prior paper

Methods evaluated by the 2022 paper: Voxel f-AnoGAN (GAN), Voxel Autoencoder
(AE), Voxel Variation Model (VM), Depth f-AnoGAN, Depth AE, Depth VM — each in
**3D-Only** and **3D+RGB** modes. Mean over the 10 categories:

### Table 2 — Anomaly localization, AUPRO @ FPR 0.3

| Method (input) | bagel | cable gland | carrot | cookie | dowel | foam | peach | potato | rope | tire | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Voxel GAN (3D only) | .440 | .453 | .825 | .755 | .782 | .378 | .392 | .639 | .775 | .389 | **.583** |
| Voxel AE (3D only) | .260 | .341 | .581 | .351 | .502 | .234 | .351 | .658 | .015 | .185 | **.348** |
| Voxel VM (3D only) | .453 | .343 | .521 | .697 | .680 | .284 | .349 | .634 | .616 | .346 | **.492** |
| Depth GAN (3D only) | .111 | .072 | .212 | .174 | .160 | .128 | .003 | .042 | .446 | .075 | **.143** |
| Depth AE (3D only) | .147 | .069 | .293 | .217 | .207 | .181 | .164 | .066 | .545 | .142 | **.203** |
| Depth VM (3D only) | .280 | .374 | .243 | .526 | .485 | .314 | .199 | .388 | .543 | .385 | **.374** |
| Voxel GAN (3D+RGB) | .664 | .620 | .766 | .740 | .783 | .332 | .582 | .790 | .633 | .483 | **.639** |
| Voxel AE (3D+RGB) | .467 | .750 | .808 | .550 | .765 | .473 | .721 | .918 | .019 | .170 | **.564** |
| Voxel VM (3D+RGB) | .510 | .331 | .413 | .715 | .680 | .279 | .300 | .507 | .611 | .366 | **.471** |
| Depth GAN (3D+RGB) | .421 | .422 | .778 | .696 | .494 | .252 | .285 | .362 | .402 | .631 | **.474** |
| Depth AE (3D+RGB) | .432 | .158 | .808 | .491 | .841 | .406 | .262 | .216 | .716 | .478 | **.481** |
| Depth VM (3D+RGB) | .388 | .321 | .194 | .570 | .408 | .282 | .244 | .349 | .268 | .331 | **.335** |

Best method reported by the paper (mean): **Voxel f-AnoGAN 3D+RGB = .639**. The
paper states performance is insufficient for practical use ("considerable room
for improvement").

### Table 5 — Anomaly detection, I-AUROC (max score)

| Method (input) | bagel | cable gland | carrot | cookie | dowel | foam | peach | potato | rope | tire | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Voxel GAN (3D only) | .383 | .623 | .474 | .639 | .564 | .409 | .617 | .427 | .663 | .577 | **.538** |
| Voxel AE (3D only) | .693 | .425 | .515 | .790 | .494 | .558 | .537 | .484 | .639 | .583 | **.572** |
| Voxel VM (3D only) | .750 | .747 | .613 | .738 | .823 | .693 | .679 | .652 | .609 | .690 | **.699** |
| Depth GAN (3D only) | .530 | .376 | .607 | .603 | .497 | .484 | .595 | .489 | .536 | .521 | **.524** |
| Depth AE (3D only) | .468 | .731 | .497 | .673 | .534 | .417 | .485 | .549 | .564 | .546 | **.546** |
| Depth VM (3D only) | .510 | .542 | .469 | .576 | .609 | .699 | .450 | .419 | .668 | .520 | **.546** |
| Voxel GAN (3D+RGB) | .680 | .324 | .565 | .399 | .497 | .482 | .566 | .579 | .601 | .482 | **.517** |
| Voxel AE (3D+RGB) | .510 | .540 | .384 | .693 | .446 | .632 | .550 | .494 | .721 | .413 | **.538** |
| Voxel VM (3D+RGB) | .553 | .772 | .484 | .701 | .751 | .578 | .480 | .466 | .689 | .611 | **.609** |
| Depth GAN (3D+RGB) | .538 | .372 | .580 | .603 | .430 | .534 | .642 | .601 | .443 | .577 | **.532** |
| Depth AE (3D+RGB) | .648 | .502 | .650 | .488 | .805 | .522 | .712 | .529 | .540 | .552 | **.595** |
| Depth VM (3D+RGB) | .513 | .551 | .477 | .581 | .617 | .716 | .450 | .421 | .598 | .623 | **.555** |

Best method reported by the paper (mean): **Voxel VM 3D-only = .699** (better
than the Voxel VM RGB + .609 and all voxel GAN/AE rows). Detection per the
paper's own benchmark is weak; meaningful gains appeared only in later
multimodal works (see §5).

## 4. Position of XMV-AD vs the base paper

- **Alignment.** XMV-AD consumes aligned RGB + Z-depth images (depth from the
  provided (x,y,z) maps) at input size 256×256 for training/eval, whereas the
  paper's benchmarks used 64³ voxel grids or 256×256 depth+RGB images. The
  depth is computed as Euclidean distance to camera center, matching the paper's
  Depth-* preprocessing philosophy.
- **Protocol adopted.** XMV-AD reports I-AUROC (max) and AUPRO with FPR limit
  0.3, exactly the paper's recommended metrics, using the official evaluation
  code protocol (implemented in `pro.py`).
- **Resolution gap.** Paper Table 2/5 evaluate at native or voxel-projected
  resolution. XMV-AD's Phase 2/3/4 tables evaluated at 256×256; Phase 5
  re-evaluates E4 at native GT resolution (`E4_concat_metrics_native.csv`) and
  records the delta vs the 256×256 numbers in
  `E4_official_metric_verification.csv`.
- **Representation fairness warning.** XMV-AD's RGB+**aligned-depth** input is
  NOT the same representation as the paper's point-cloud voxel grids (nor
  M3DM's). Pixel-level scores and AU-PRO are tied to the input representation;
  table cells are therefore **not** numerically interchangeable across
  representations. We never claim superiority over the paper's voxel baselines
  or over M3DM on that basis.

## 5. Later comparator — M3DM (Wang et al., CVPR 2023)

M3DM is the widely-cited later multimodal baseline (point-cloud + RGB, hybrid
feature/decision fusion). Numbers below are **Reported by prior paper**:

| Setting | I-AUROC (mean) | AUPRO (mean) | Source |
|---|---|---|---|
| M3DM 3D-only | **.874** | — | CVPR 2023 paper |
| M3DM RGB-only task | — | **.942** | CVPR 2023 paper |
| M3DM RGB+3D (full) | **.945** | **.964** | CVPR 2023 paper |

M3DM's per-category I-AUROC (RGB+3D): bagel .994, cable gland .909, carrot
.972, cookie .976, dowel .960, foam .942, peach .973, potato .899, rope .972,
tire .850 (mean .945).

XMV-AD explicitly does **not** position itself against M3DM: M3DM uses
pre-trained ViT feature extraction + point-cloud memory banks (GPU-trained,
>50 GB budget) — a different hardware regime and representation. Where M3DM is
tabulated for orientation (Phase 5 baseline table), it carries the provenance
label "Reported by prior paper (dispatchless bounds — not directly comparable)".