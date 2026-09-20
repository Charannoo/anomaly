"""E3B reliability-aware score fusion (Phase 3B, pre-registered rules only).

Reads frozen Phase 2 test predictions + frozen E3 calibration +
E3B validation-normal reliability stats. No retraining, no test tuning.

Rules (see src/xmvad/fusion/score_fusion.py for exact formulas):
  E3B-1 static:      per-category base weights from normal-state noise.
  E3B-2 validity:    E3B-1 base x per-image depth-valid fraction.
  E3B-3 evidence:    per-image softmax over calibrated evidence, fixed T=1.
Oracle (TEST-ORACLE, analysis only, isolated module) + error-correlation
analysis included. Outputs tables under experiments/tables/ (E3B_*).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from xmvad.analysis.oracle import oracle_image_scores, oracle_map_set
from xmvad.data.depth import load_xyz, valid_mask_from_xyz
from xmvad.fusion.score_fusion import (
    TEMPERATURE,
    calibrate,
    evidence_weights,
    fuse_maps,
    static_weights,
    validity_adjusted_weights,
)
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores

METHODS = ("static", "validity", "evidence")


def load_pred_set(pred_dir: Path, category: str) -> dict[str, dict]:
    out = {}
    for f in sorted(pred_dir.glob("*.npz")):
        rest = f.stem[len(category) + 1:]
        parts = rest.split("_")
        split, filestem = parts[0], parts[-1]
        defect = "_".join(parts[1:-1])
        sid = f"{category}/{split}/{defect}/{filestem}"
        z = np.load(f, allow_pickle=True)
        out[sid] = {"map": z["map"].astype(np.float32), "gt": z["gt"].astype(bool),
                    "label": int(z["label"]), "defect": str(z["defect"])}
    return out


def xyz_path_for(root: Path, sample_id: str) -> Path:
    parts = sample_id.split("/")
    xyz_dir = root / parts[0] / parts[1] / parts[-2] / "xyz"
    for ext in (".tiff", ".tif"):
        p = xyz_dir / f"{parts[-1]}{ext}"
        if p.exists():
            return p
    cands = sorted(xyz_dir.glob(parts[-1] + ".tif*"))
    if not cands:
        raise FileNotFoundError(f"XYZ backing not found for {sample_id}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-preds", default="experiments/predictions/E1_baseline_rgb")
    ap.add_argument("--depth-preds", default="experiments/predictions/E2_baseline_depth")
    ap.add_argument("--calibration", default="experiments/calibration/E3_late_fusion_calibration.json")
    ap.add_argument("--reliability", default="experiments/calibration/E3B_reliability_stats.json")
    ap.add_argument("--data", default="data/raw/mvtec3d")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    with open(REPO_ROOT / args.reliability, encoding="utf-8") as f:
        rel = json.load(f)
    root = REPO_ROOT / args.data
    cats = args.categories or sorted(calib["categories"])

    metric_rows, defect_rows, corr_rows = [], [], []
    oracle_img, oracle_loc = {}, {}
    for cat in cats:
        print(f"== {cat} ==", flush=True)
        rgb_set = load_pred_set(REPO_ROOT / args.rgb_preds / cat, cat)
        depth_set = load_pred_set(REPO_ROOT / args.depth_preds / cat, cat)
        if set(rgb_set) != set(depth_set):
            raise ValueError(f"[{cat}] prediction ID mismatch")
        sids = sorted(rgb_set)
        cm = calib["categories"][cat]
        base_w = rel["categories"][cat]["static_weights"]
        maps = {m: [] for m in METHODS}
        gts, labels, defects = [], [], []
        rgb_sc, depth_sc = [], []
        for sid in tqdm(sids, desc=f"fuse {cat}", leave=False):
            r, d = rgb_set[sid], depth_set[sid]
            if r["map"].shape != d["map"].shape or r["gt"].shape != d["gt"].shape \
                    or not np.array_equal(r["gt"], d["gt"]) \
                    or r["label"] != d["label"] or r["defect"] != d["defect"]:
                raise ValueError(f"[{sid}] pairing mismatch")
            v = np.array(Image.fromarray(
                (valid_mask_from_xyz(load_xyz(xyz_path_for(root, sid))).astype(np.uint8)) * 255
            ).resize((r["map"].shape[1], r["map"].shape[0]), Image.NEAREST)) > 127
            zr = calibrate(r["map"], cm["rgb"]["median"], cm["rgb"]["scale"])
            zd = calibrate(d["map"], cm["depth"]["median"], cm["depth"]["scale"])
            q = float(v.mean())
            w_s = (base_w["w_rgb"], base_w["w_depth"])
            w_v = validity_adjusted_weights(base_w["w_rgb"], base_w["w_depth"], q)
            e_r = max(0.0, float(np.max(zr[np.isfinite(zr)])) if np.isfinite(zr).any() else 0.0)
            ev = zd[v] if v.any() else np.array([0.0])
            e_d = max(0.0, float(np.max(ev[np.isfinite(ev)])) if np.isfinite(ev).any() else 0.0)
            w_e = evidence_weights(e_r, e_d, TEMPERATURE)
            maps["static"].append(fuse_maps(zr, zd, v, *w_s))
            maps["validity"].append(fuse_maps(zr, zd, v, *w_v))
            maps["evidence"].append(fuse_maps(zr, zd, v, *w_e))
            gts.append(r["gt"])
            labels.append(r["label"])
            defects.append(r["defect"])
            rgb_sc.append(float(np.max(r["map"])) if np.isfinite(r["map"]).any() else 0.0)
            depth_sc.append(float(np.max(d["map"])) if np.isfinite(d["map"]).any() else 0.0)
        labels_np = np.array(labels)
        rgb_sc, depth_sc = np.array(rgb_sc), np.array(depth_sc)

        row: dict = {"category": cat}
        for m in METHODS:
            sc = {name: np.array([variant_scores(a, None)[name] for a in maps[m]])
                  for name, _ in SCORE_VARIANTS}
            row[f"{m}_image_auroc_max"] = image_auroc(labels_np, sc["max"])
            row[f"{m}_image_auprc_max"] = image_auprc(labels_np, sc["max"])
            row[f"{m}_image_auroc_topk1"] = image_auroc(labels_np, sc["topk_1pct"])
            row[f"{m}_pixel_auroc"] = pixel_auroc(gts, maps[m])
            row[f"{m}_pixel_auprc"] = pixel_auprc(gts, maps[m])
            row[f"{m}_aupro"] = aupro(gts, maps[m])
            row[f"_scores_{m}_max"] = sc["max"]  # kept for oracle/corr below
        metric_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

        # Oracle diagnostics (TEST-ORACLE, analysis only).
        o_scores = oracle_image_scores(row["_scores_static_max"] * 0 + rgb_sc, depth_sc, labels_np)
        # NOTE: oracle uses RAW branch max scores (same readout both branches);
        # calibrated variants would rescale monotonically per branch — ranking
        # within branch is what matters for selection headroom. Documented.
        from sklearn.metrics import roc_auc_score

        oracle_img[cat] = float(roc_auc_score(labels_np, o_scores))
        o_maps = oracle_map_set(
            [rgb_set[s]["map"] for s in sids], [depth_set[s]["map"] for s in sids],
            gts, labels_np)
        oracle_loc[cat] = {"pixel_auroc": pixel_auroc(gts, o_maps), "aupro": aupro(gts, o_maps)}

        # Error correlation (raw branch max scores).
        from scipy.stats import pearsonr, spearmanr

        for tag, mask in (("normal", labels_np == 0), ("anomalous", labels_np == 1)):
            x, y = rgb_sc[mask], depth_sc[mask]
            corr_rows.append({"category": cat, "subset": tag, "n": int(mask.sum()),
                              "pearson": round(float(pearsonr(x, y)[0]), 4),
                              "spearman": round(float(spearmanr(x, y)[0]), 4)})

        # Defect-level AUROC per rule (max variant).
        normal_idx = [i for i, l in enumerate(labels) if l == 0]
        by_defect: dict[str, list[int]] = {}
        for i, p in enumerate(sids):
            if labels[i] == 1:
                by_defect.setdefault(defects[i], []).append(i)
        for dft, idxs in sorted(by_defect.items()):
            entry = {"category": cat, "defect": dft, "n_normal": len(normal_idx), "n_defect": len(idxs)}
            for m in METHODS:
                s = np.array([variant_scores(a, None)["max"] for a in maps[m]])
                y = np.array([0] * len(normal_idx) + [1] * len(idxs))
                from sklearn.metrics import roc_auc_score as ras

                entry[f"{m}_auroc"] = round(float(ras(y, np.concatenate([s[normal_idx], s[idxs]]))), 4)
            defect_rows.append(entry)

    tables = REPO_ROOT / "experiments" / "tables"
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables / "E3B_reliability_metrics.csv", index=False)

    # Comparison vs RGB / Depth / fixed-mean (frozen Phase 2 + 3A tables).
    rgb_df = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    depth_df = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    fixed_df = pd.read_csv(tables / "E3_vs_baselines.csv")
    comp = []
    for _, fr in metrics_df.iterrows():
        cat = fr["category"]
        rr = rgb_df[rgb_df.category == cat].iloc[0]
        dr = depth_df[depth_df.category == cat].iloc[0]
        base = {"I-AUROC(max)": (float(rr["image_auroc_max"]), float(dr["image_auroc_max"])),
                "I-AUROC(topk1%)": (float(rr["image_auroc_topk_1pct"]), float(dr["image_auroc_topk_1pct"])),
                "P-AUROC": (float(rr["pixel_auroc"]), float(dr["pixel_auroc"])),
                "AUPRO": (float(rr["aupro"]), float(dr["aupro"]))}
        for metric, (rv, dv) in base.items():
            best = max(rv, dv)
            for rule, col in (("Static", "static"), ("Validity", "validity"), ("Evidence", "evidence")):
                key = {"I-AUROC(max)": f"{col}_image_auroc_max",
                       "I-AUROC(topk1%)": f"{col}_image_auroc_topk1",
                       "P-AUROC": f"{col}_pixel_auroc",
                       "AUPRO": f"{col}_aupro"}[metric]
                fv = float(fr[key])
                comp.append({"category": cat, "metric": metric, "rule": rule,
                             "RGB": round(rv, 4), "Depth": round(dv, 4),
                             "FixedMean": round(float(fixed_df[(fixed_df.category == cat)
                                                               & (fixed_df.metric == metric)]["LateFusion"].iloc[0]), 4),
                             "Rule": round(fv, 4), "d_vs_BestSingle": round(fv - best, 4),
                             "d_vs_FixedMean": round(fv - float(fixed_df[(fixed_df.category == cat)
                                                                          & (fixed_df.metric == metric)]["LateFusion"].iloc[0]), 4),
                             "oracle_I-AUROC": round(oracle_img[cat], 4) if metric == "I-AUROC(max)" else ""})
    # MEAN rows
    for metric in ("I-AUROC(max)", "I-AUROC(topk1%)", "P-AUROC", "AUPRO"):
        key = {"I-AUROC(max)": "image_auroc_max", "I-AUROC(topk1%)": "image_auroc_topk_1pct",
               "P-AUROC": "pixel_auroc", "AUPRO": "aupro"}[metric]
        rkey = {"I-AUROC(max)": "static_image_auroc_max", "I-AUROC(topk1%)": "static_image_auroc_topk1",
                "P-AUROC": "static_pixel_auroc", "AUPRO": "static_aupro"}[metric]
        rv = float(np.nanmean(rgb_df[rgb_df.category != "MEAN"][key].to_numpy(float)))
        dv = float(np.nanmean(depth_df[depth_df.category != "MEAN"][key].to_numpy(float)))
        for rule, col in (("Static", "static"), ("Validity", "validity"), ("Evidence", "evidence")):
            ck = rkey.replace("static", col)
            fv = float(np.nanmean(metrics_df[ck].to_numpy(float)))
            fxf = float(np.nanmean(fixed_df[(fixed_df.category != "MEAN") & (fixed_df.metric == metric)]["LateFusion"].to_numpy(float)))
            comp.append({"category": "MEAN", "metric": metric, "rule": rule,
                         "RGB": round(rv, 4), "Depth": round(dv, 4),
                         "FixedMean": round(fxf, 4), "Rule": round(fv, 4),
                         "d_vs_BestSingle": round(fv - max(rv, dv), 4),
                         "d_vs_FixedMean": round(fv - fxf, 4),
                         "oracle_I-AUROC": round(float(np.mean(list(oracle_img.values()))), 4) if metric == "I-AUROC(max)" else ""})
    pd.DataFrame(comp).to_csv(tables / "E3B_vs_all.csv", index=False)
    oracle_loc_rows = [{"category": c, **{k: round(float(v), 4) for k, v in m.items()},
                        "oracle_image_auroc": round(float(oracle_img[c]), 4)} for c, m in oracle_loc.items()]
    pd.DataFrame(oracle_loc_rows).to_csv(tables / "E3B_oracle.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(tables / "E3B_defect_level.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(tables / "E3B_error_correlation.csv", index=False)
    print("wrote E3B tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
