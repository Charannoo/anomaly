# Dataset (MVTec 3D-AD)

## Source

Official MVTec 3D-AD dataset page. Extract/copy to `data/raw/mvtec3d/`.
Kaggle mirrors only with explicit approval. Dataset files are never committed.

## Expected layout

```
data/raw/mvtec3d/<category>/{train,validation,test}/...
train/validation: good/rgb/*.png + good/xyz/*.tiff
test: good/... + <defect>/rgb + <defect>/xyz + <defect>/gt
```

Categories (10): bagel, cable_gland, carrot, cookie, dowel, foam, peach,
potato, rope, tire. Discovery is dynamic — names are never hard-coded.

## Modalities

- RGB: `rgb/*.png` (PIL RGB).
- 3D: `xyz/*.tiff` organized XYZ (H x W x 3) via tifffile; raw files kept for
  future RGB+XYZ experiments.
- Depth (v1): `Z = XYZ[..., 2]`; invalid sentinel `(0,0,0)` + non-finite tracked
  by an explicit valid mask, never silently converted to depth.
- GT: `<defect>/gt/*.png` (>0 = anomalous), resized NEAREST to RGB size.

## Preprocessing contract

- Shared resize to `input_size` (default 256x256): RGB bilinear, depth/valid/GT
  nearest; identical geometric params for all modalities+masks.
- RGB: ImageNet mean/std; color jitter RGB-only (train).
- Depth: robust stats fit on TRAIN-NORMAL only (`robust_median` default:
  `(z-median)/MAD-scale`, clip [-3,3]; `percentile` alternative configurable);
  invalid -> 0.0 + mask; depth noise depth-only (train).
- Raw never modified; outputs under `data/processed/`; caches under `data/cache/`.

## Validation

`python scripts/inspect_dataset.py --data data/raw/mvtec3d` reports split
counts, pairing gaps, shapes, depth min/max, invalid %, GT/alignment warnings.
`python scripts/visualize_sample.py ...` renders 25/category RGB/depth/edge
grids; do not start fusion work with unresolved registration errors.
