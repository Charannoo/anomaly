# XMV-AD-H Colab Bundle

High-accuracy anomaly detection on **MVTec-3D-AD** (10 categories) with
DINO ViT-B/8 (RGB) + Point-MAE (XYZ) features, compared against the published
**M3DM** numbers.

The whole pipeline (feature extraction, H1/H2/H3 scoring) runs on a stock
Google-Colab GPU notebook. The Point-MAE backbone was rebuilt in pure PyTorch
(grouping = masked FPS + masked KNN) so the ancient `pointnet2_ops` / `knn_cuda`
CUDA extensions are **never** compiled on Colab; it loads the same official
`pointmae_pretrain.pth` weights.

## What is inside

| File | Purpose |
| --- | --- |
| `XMV_AD_H_FAST.ipynb` | 5-cell Colab notebook |
| `bootstrap.py` | locate/extract dataset, download+verify checkpoint, restore Drive caches |
| `run_all.py` | end-to-end driver (extract -> H1 -> H2 -> H3 -> comparison) |
| `extract_features.py` | DINO + Point-MAE feature extraction with resumable shard cache |
| `backbones.py` | pure-PyTorch PointTransformer + DINO wrapper + cloud I/O |
| `cache_utils.py` | shared config/paths, shard manifest, coreset, exact-NN scoring |
| `metrics.py` | I-AUROC, P-AUROC, AUPRO@0.3 (official MVTec code) |
| `run_h1.py` | H1: RGB memory bank -> `results/H1_RGB.csv` |
| `run_h2.py` | H2: XYZ memory bank -> `results/H2_XYZ.csv` |
| `run_h3.py` | H3A 50/50, H3B max, H3C concat -> `results/H3_multimodal.csv` |
| `status.py` | summary + comparison vs published M3DM |
| `config_colab.yaml` | all settings (paths, backbones, coreset, published M3DM) |
| `requirements_colab.txt` | Python deps |

## Quick start (Colab)

1. Put the dataset archive in Drive:
   `MyDrive/xmvad-ha/dataset/mvtec3d.tar.gz` (`.tar.zst` / `.tar.xz` also work).
2. Upload this bundle as `MyDrive/xmvad-ha/xmvad_colab_bundle.zip`
   (or `xmvad_colab_bundle.zip` at `/content`).
3. `Runtime -> Change runtime type -> GPU`.
4. Open `XMV_AD_H_FAST.ipynb` and `Runtime -> Run all`.

The notebook mounts Drive, unpacks the bundle to `/content/xmvad`, installs the
tiny requirements, then runs the pipeline. Output is **visible in cell 4** while
a copy is tee'd to `MyDrive/xmvad-ha/logs/run_all.log`.

## What runs

- **bootstrap** – finds and extracts the archive under `/content/xmvad_work/data`,
  downloads the Point-MAE checkpoint (size + sha verified), and copies any
  previously computed feature shards from Drive so re-runs are instant.
- **extract** – per category/split, DINO `rgb` (784x768 @ 28x28) and
  Point-MAE `xyz` (3136x1152 @ 56x56) shards
  (`features/<cat>_<split>_<modality>.pt` + `cache_manifest.json`). Cached
  shards print `[CACHE HIT] <cat> <split> <modality>` and are skipped.
- **H1 / H2** – PatchCore-style memory banks: coreset (`f_coreset=0.1`,
  `eps=0.9`, exact-L2 NN) + reweighted score and 224x224 anomaly map.
- **H3** – `H3A` calibrated 50/50 fusion, `H3B` max fusion, `H3C` aligned
  concat bank (RGB up-sampled to the XYZ 56x56 grid with the repo's canonical
  repeat layout; 1920-D per cell).
- **compare** – `results/M3DM_comparison.csv` with per-category best-of-ours
  vs published M3DM and gaps.

## Provenance labelling

- M3DM numbers (`I` 0.945, `P` 0.992, `AUPRO` 0.964) are from the paper and are
  labelled **`reported_prior_work`** everywhere (`config_colab.yaml` +
  `M3DM_comparison.csv`).
- Everything we measure is labelled **`measured_by_us`**.

## Outputs

- `results/H1_RGB.csv`, `results/H2_XYZ.csv`, `results/H3_multimodal.csv`,
  `results/M3DM_comparison.csv` (also mirrored to
  `MyDrive/xmvad-ha/results/`).
- `logs/run_all.log` on Drive.
- `features/` shards on both the instance and Drive (resume-friendly).

## Running locally (no Colab)

`cache_utils.load_config()` rewrites all paths under the bundle directory when
`/content` is absent:

```
pip install -r requirements_colab.txt
python bootstrap.py          # dataset archive must be in work_local/data/
python run_all.py
```

## Notes

- Dataset must follow the canonical layout inside the archive:
  `<category>/{train/good, test/...}/{rgb,xyz,gt}`.
- `include_combined: true` also takes `test/combined/` as test samples (matches
  the M3DM loader protocol).
- To change batch sizes or the cache precision, edit `config_colab.yaml`.