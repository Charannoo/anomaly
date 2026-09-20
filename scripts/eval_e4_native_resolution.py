"""Phase 5, todo 3: re-evaluate frozen E4 at NATIVE MVTec GT resolution.

Per category, the already-computed E4 test maps (256x256) are resized to the
original GT dimensions with BILINEAR interpolation; NO thresholding happens
before or after the resize. Native-resolution GT masks are re-loaded from the
raw dataset (test/.../gt/*.png). Image-level scores are recomputed on the
256x256 fused maps (resolution-invariant: max / top-k) and are therefore
identical to Phase 4A; pixel-level metrics (pixel-AUROC/AUPRC, AUPRO) are
recomputed at native resolution against native GT.

Official protocol note: the official MVTec evaluation code compares
FULL-resolution maps. Phase 2/3/4 evaluated at 256x256; this script closes that
gap. See docs/BASE_PAPER_2022.md §4.

Writes experiments/tables/E4_concat_metrics_native.csv and a 256-vs-native
delta table E4_official_metric_verification.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import gc

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from xmvad.data.mvtec3d import discover_categories, discover_split, load_gt_mask
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro_stream
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.utils.config import load_config


def load_config_root(config: str = "configs/evaluate_e4.yaml") -> str:
    """Small helper: config path is optional; fall back to concat config."""
    p = Path(config)
    if p.exists():
        return str(Path(load_config(config).get("data", {}).get("root",
                                                                 "data/raw/mvtec3d")))
    return "data/raw/mvtec3d"


def load_native_gt(record) -> np.ndarray:
    """Native-resolution boolean GT; all-false mask when absent (normal)."""
    gt = load_gt_mask(record.gt_path)
    if gt is None:
        rgb = Path(record.rgb_path)
        img = Image.open(rgb)
        return np.zeros((img.size[1], img.size[0]), dtype=bool)
    return gt


def upscale(map256: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Bilinear upscale WITHOUT thresholding (official full-res protocol)."""
    arr = np.asarray(map256, dtype=np.float32)
    img = Image.fromarray(arr, mode="F").resize(wh, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def main() -> int:
    cfg_path = "configs/concat_fusion.yaml"
    cfg = load_config(cfg_path)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    cats = discover_categories(str(root))
    pred_dir = REPO_ROOT / "experiments" / "predictions" / "E4_concat_fusion"

    rows_native: list[dict] = []
    for cat in tqdm(cats, desc="native re-eval", leave=False):
        recs = discover_split(root, cat, "test")
        gts_256 = []
        gts_nat: list[np.ndarray] = []
        maps_mean_256 = []
        maps_mean_nat: list[np.ndarray] = []
        maps_max_256 = []
        maps_max_nat: list[np.ndarray] = []
        labels, defects, ids = [], [], []
        native_wh: tuple[int, int] | None = None
        for r in recs:
            sid = r.sample_id
            npz_path = pred_dir / cat / f"{sid.replace('/', '_')}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"missing E4 predictions for {sid}")
            z = np.load(npz_path)
            gt256 = np.asarray(z["gt"], dtype=bool)
            mm = np.asarray(z["map_mean"], dtype=np.float64)
            mx = np.asarray(z["map_max"], dtype=np.float64)
            gt_nat = load_native_gt(r)
            if native_wh is None:
                native_wh = (gt_nat.shape[1], gt_nat.shape[0])
            elif (gt_nat.shape[1], gt_nat.shape[0]) != native_wh:
                raise ValueError(f"mixed native resolutions in {cat}")
            gts_256.append(gt256)
            gts_nat.append(gt_nat)
            maps_mean_256.append(mm)
            maps_mean_nat.append(upscale(mm, native_wh))
            maps_max_256.append(mx)
            maps_max_nat.append(upscale(mx, native_wh))
            labels.append(r.label)
            defects.append(r.defect)
            ids.append(sid)
        labels_np = np.array(labels)
        sc_mean = {n: np.array([variant_scores(m, None)[n] for m in maps_mean_256])
                   for n, _ in SCORE_VARIANTS}
        del gts_256, maps_mean_256, maps_max_256, mm, mx, gt256  # type: ignore[name-defined]
        gc.collect()

        def row(label: str, gts: list, maps: list, score_key: str, i_auroc: np.ndarray):
            ia = float(i_auroc)
            iauprc = image_auprc(labels_np, sc_mean["max"])
            i1 = image_auroc(labels_np, sc_mean["topk_1pct"])
            pa = pixel_auroc(gts, maps)
            gc.collect()
            pa_prc = pixel_auprc(gts, maps)
            gc.collect()
            ap = aupro_stream(gts, maps)
            gc.collect()
            return {
                "category": cat,
                "res": label,
                "native_wh": f"{native_wh[0]}x{native_wh[1]}",
                "image_auroc_max": ia,
                "image_auprc_max": iauprc,
                "image_auroc_topk1": i1,
                "pixel_auroc": pa,
                "pixel_auprc": pa_prc,
                "aupro": ap,
            }

        iam = image_auroc(labels_np, sc_mean["max"])
        rows_native.append(row("native", gts_nat, maps_mean_nat, "map_mean", iam))
        del maps_mean_nat
        gc.collect()
        rows_native.append(row("native_max", gts_nat, maps_max_nat, "map_max", iam))
        del maps_max_nat, gts_nat
        gc.collect()

    native_df = pd.DataFrame(rows_native)
    tables = REPO_ROOT / "experiments" / "tables"
    native_df.to_csv(tables / "E4_concat_metrics_native.csv", index=False)

    # Verify vs 256-resolution Phase 4A table.
    p4 = pd.read_csv(tables / "E4_concat_metrics.csv").set_index("category")
    p5 = native_df[native_df.res == "native"].set_index("category")
    verify = []
    for cat in cats:
        if cat not in p4.index or cat not in p5.index:
            continue
        verify.append({
            "category": cat,
            "image_auroc_256": p4.loc[cat, "image_auroc_max"],
            "image_auroc_native": p5.loc[cat, "image_auroc_max"],
            "d_image_auroc": float(p5.loc[cat, "image_auroc_max"]) - float(p4.loc[cat, "image_auroc_max"]),
            "pixel_auroc_256": p4.loc[cat, "pixel_auroc"],
            "pixel_auroc_native": p5.loc[cat, "pixel_auroc"],
            "d_pixel_auroc": float(p5.loc[cat, "pixel_auroc"]) - float(p4.loc[cat, "pixel_auroc"]),
            "aupro_256": p4.loc[cat, "aupro"],
            "aupro_native": p5.loc[cat, "aupro"],
            "d_aupro": float(p5.loc[cat, "aupro"]) - float(p4.loc[cat, "aupro"]),
        })
    p4m = p4[p4.index != "MEAN"].astype(float) if "MEAN" in p4.index else p4.astype(float)
    p5m = p5[p5.index != "MEAN"] if "MEAN" in p5.index else p5
    verify.append({
        "category": "MEAN",
        "image_auroc_256": round(float(p4m["image_auroc_max"].mean()), 4),
        "image_auroc_native": round(float(p5m["image_auroc_max"].mean()), 4),
        "d_image_auroc": round(float(p5m["image_auroc_max"].mean()) - float(p4m["image_auroc_max"].mean()), 4),
        "pixel_auroc_256": round(float(p4m["pixel_auroc"].mean()), 4),
        "pixel_auroc_native": round(float(p5m["pixel_auroc"].mean()), 4),
        "d_pixel_auroc": round(float(p5m["pixel_auroc"].mean()) - float(p4m["pixel_auroc"].mean()), 4),
        "aupro_256": round(float(p4m["aupro"].mean()), 4),
        "aupro_native": round(float(p5m["aupro"].mean()), 4),
        "d_aupro": round(float(p5m["aupro"].mean()) - float(p4m["aupro"].mean()), 4),
    })
    ver_df = pd.DataFrame(verify)
    ver_df.to_csv(tables / "E4_official_metric_verification.csv", index=False)

    print(f"wrote {tables / 'E4_concat_metrics_native.csv'} and "
          f"{tables / 'E4_official_metric_verification.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())