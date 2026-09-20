"""E3C real-test inference — RUN ONCE on frozen MVTec 3D-AD test predictions.

Gate and all parameters are frozen before this script runs. No retraining
after inspecting results. Uses saved Phase 2 test maps + raw validity masks.

Detection:  S_final = w_img_rgb*S_rgb_cal + w_img_depth*S_depth_cal
Localization: A_final = w_map_rgb*Zr + w_map_depth*Zd (invalid -> RGB only)

Outputs: predictions/E3C_gate/, tables/E3C_{metrics,vs_all,defect_level,
weights,collapse}.csv, figures/gate/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from xmvad.data.depth import depth_to_uint8, extract_depth, load_xyz, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_split
from xmvad.fusion.gate_features import gate_features
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.models.fusion.reliability_gate import ReliabilityGate


def load_pred_set(pred_dir: Path, category: str) -> dict[str, dict]:
    out = {}
    for f in sorted(pred_dir.glob("*.npz")):
        rest = f.stem[len(category) + 1:]
        parts = rest.split("_")
        sid = f"{category}/{parts[0]}/{'_'.join(parts[1:-1])}/{parts[-1]}"
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
        raise FileNotFoundError(sample_id)
    return cands[0]


def save_gate_fig(path: Path, rgb_u8, depth_u8, gt, rgb_map, depth_map, fused,
                  w_img, w_map, title):
    import matplotlib.cm as cm

    def overlay(base, m):
        m = np.clip(m, 0, 1)
        col = (cm.jet(m)[..., :3] * 255).astype(np.uint8)
        return (0.55 * base.astype(float) + 0.45 * col.astype(float)).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"{title}\nw_img=({w_img[0]:.2f},{w_img[1]:.2f}) "
                 f"w_map=({w_map[0]:.2f},{w_map[1]:.2f})", fontsize=9)
    axes[0, 0].imshow(rgb_u8)
    axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(depth_u8, cmap="gray")
    axes[0, 1].set_title("depth")
    axes[0, 2].imshow(gt.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("GT")
    axes[1, 0].imshow(np.clip(rgb_map, 0, 1), cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("RGB map")
    axes[1, 1].imshow(np.clip(depth_map, 0, 1), cmap="jet", vmin=0, vmax=1)
    axes[1, 1].set_title("Depth map")
    axes[1, 2].imshow(overlay(rgb_u8, fused / (np.abs(fused).max() + 1e-9) * 0.5 + 0.25))
    axes[1, 2].set_title("E3C fused overlay")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-preds", default="experiments/predictions/E1_baseline_rgb")
    ap.add_argument("--depth-preds", default="experiments/predictions/E2_baseline_depth")
    ap.add_argument("--calibration", default="experiments/calibration/E3_late_fusion_calibration.json")
    ap.add_argument("--gate-dir", default="checkpoints/E3C_gate")
    ap.add_argument("--gate-data", default="experiments/gate_data")
    ap.add_argument("--data", default="data/raw/mvtec3d")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    gate_cfg = json.load(open(REPO_ROOT / args.gate_dir / "config.json", encoding="utf-8"))
    norm = json.load(open(REPO_ROOT / args.gate_dir / "feat_norm.json", encoding="utf-8"))
    mu, sd = np.array(norm["mean"]), np.array(norm["std"])
    gate = ReliabilityGate(in_dim=len(mu))
    gate.load_state_dict(torch.load(REPO_ROOT / args.gate_dir / "gate_best.pt",
                                    map_location="cpu", weights_only=True)["model_state"])
    gate.eval()
    root = REPO_ROOT / args.data
    cats = args.categories or sorted(calib["categories"])
    out_pred = REPO_ROOT / "experiments" / "predictions" / "E3C_gate"
    fig_base = REPO_ROOT / "experiments" / "figures" / "gate"

    metric_rows, defect_rows, weight_rows = [], [], []
    for cat in cats:
        print(f"== {cat} ==", flush=True)
        rgb_set = load_pred_set(REPO_ROOT / args.rgb_preds / cat, cat)
        depth_set = load_pred_set(REPO_ROOT / args.depth_preds / cat, cat)
        if set(rgb_set) != set(depth_set):
            raise ValueError(f"[{cat}] prediction ID mismatch")
        sids = sorted(rgb_set)
        cm = calib["categories"][cat]
        z = np.load(REPO_ROOT / args.gate_data / f"{cat}.npz")
        r_med, r_p99, d_med, d_p99 = (float(v) for v in z["val_score_dist"])
        (out_pred / cat).mkdir(parents=True, exist_ok=True)
        (fig_base / cat).mkdir(parents=True, exist_ok=True)
        fmaps, gts, labels, defects, s_fin = [], [], [], [], []
        w_img_all, w_map_all = [], []
        thr = cm["score_thresholds"]["fused_max_p95"]  # operating convention, documented
        for sid in tqdm(sids, desc=f"gate {cat}", leave=False):
            r, d = rgb_set[sid], depth_set[sid]
            if r["map"].shape != d["map"].shape or r["gt"].shape != d["gt"].shape \
                    or not np.array_equal(r["gt"], d["gt"]) \
                    or r["label"] != d["label"] or r["defect"] != d["defect"]:
                raise ValueError(f"[{sid}] pairing mismatch")
            H, W = r["map"].shape
            v = np.array(Image.fromarray(
                (valid_mask_from_xyz(load_xyz(xyz_path_for(root, sid))).astype(np.uint8)) * 255
            ).resize((W, H), Image.NEAREST)) > 127
            zr = calibrate(r["map"], cm["rgb"]["median"], cm["rgb"]["scale"])
            zd = calibrate(d["map"], cm["depth"]["median"], cm["depth"]["scale"])
            feats = np.array(gate_features(zr, zd, v), dtype=np.float32)
            with torch.no_grad():
                w = gate(torch.from_numpy((feats - mu) / sd).unsqueeze(0).float())
            wi = w["w_img"][0].numpy()
            wm = w["w_map"][0].numpy()
            w_img_all.append(wi)
            w_map_all.append(wm)
            sr = (float(np.max(r["map"])) - r_med) / (r_p99 - r_med + 1e-9)
            evd = d["map"][v] if v.any() else np.array([0.0])
            sd = (float(np.max(evd)) - d_med) / (d_p99 - d_med + 1e-9)
            s_fin.append(float(wi[0] * sr + wi[1] * sd))
            fused = np.where(v, wm[0] * zr + wm[1] * zd, zr)  # invalid -> RGB only
            fused = np.where(np.isfinite(fused), fused, 0.0).astype(np.float32)
            fmaps.append(fused)
            gts.append(r["gt"])
            labels.append(r["label"])
            defects.append(r["defect"])
            np.savez_compressed(out_pred / cat / f"{sid.replace('/', '_')}.npz",
                                map=fused, gt=r["gt"], label=np.int64(r["label"]),
                                defect=r["defect"], sample_id=sid,
                                score=np.float32(s_fin[-1]),
                                w_img=wi.astype(np.float32), w_map=wm.astype(np.float32))
        labels_np = np.array(labels)
        s_fin = np.array(s_fin)
        sc = {name: np.array([variant_scores(m, None)[name] for m in fmaps])
              for name, _ in SCORE_VARIANTS}
        row = {"category": cat,
               "image_auroc_max": image_auroc(labels_np, s_fin),
               "image_auprc_max": image_auprc(labels_np, s_fin),
               "image_auroc_topk1": image_auroc(labels_np, sc["topk_1pct"]),
               "pixel_auroc": pixel_auroc(gts, fmaps),
               "pixel_auprc": pixel_auprc(gts, fmaps),
               "aupro": aupro(gts, fmaps),
               "num_test_images": len(sids), "num_anomalous": int(labels_np.sum())}
        metric_rows.append(row)
        # Defect-level + fixed/broken counts (validation-p95 operating point).
        normal_idx = [i for i, l in enumerate(labels) if l == 0]
        by_defect: dict[str, list[int]] = {}
        for i, s in enumerate(sids):
            if labels[i] == 1:
                by_defect.setdefault(defects[i], []).append(i)
        for dft, idxs in sorted(by_defect.items()):
            y = np.array([0] * len(normal_idx) + [1] * len(idxs))
            from sklearn.metrics import roc_auc_score as ras
            defect_rows.append({"category": cat, "defect": dft, "n_normal": len(normal_idx),
                                "n_defect": len(idxs),
                                "e3c_auroc": round(float(ras(y, np.concatenate([s_fin[normal_idx], s_fin[idxs]]))), 4)})
        w_img_all = np.array(w_img_all)
        w_map_all = np.array(w_map_all)
        for name, arr in (("w_img_rgb", w_img_all[:, 0]), ("w_img_depth", w_img_all[:, 1]),
                          ("w_map_rgb", w_map_all[:, 0]), ("w_map_depth", w_map_all[:, 1])):
            weight_rows.append({"category": cat, "weight": name,
                                "mean": round(float(arr.mean()), 4),
                                "std": round(float(arr.std()), 4),
                                "p5": round(float(np.percentile(arr, 5)), 4),
                                "p50": round(float(np.percentile(arr, 50)), 4),
                                "p95": round(float(np.percentile(arr, 95)), 4)})
        weight_rows.append({"category": cat, "weight": "mean_abs_img_minus_map_rgb",
                            "mean": round(float(np.abs(w_img_all[:, 0] - w_map_all[:, 0]).mean()), 4),
                            "std": "", "p5": "", "p50": "", "p95": ""})
        # Weight example figures (anomalous only).
        anom = [i for i, l in enumerate(labels) if l == 1]
        picks = {}
        if anom:
            picks["rgb_dom"] = max(anom, key=lambda i: w_map_all[i, 0])
            picks["depth_dom"] = max(anom, key=lambda i: w_map_all[i, 1])
            picks["balanced"] = min(anom, key=lambda i: abs(w_map_all[i, 0] - 0.5))
        for tag, i in picks.items():
            p_rgb = rgb_set[sids[i]]["map"]
            p_dep = depth_set[sids[i]]["map"]
            rec = next(x for x in discover_split(root, cat, sids[i].split("/")[1])
                       if x.sample_id == sids[i])
            rgb_u8 = np.array(Image.open(rec.rgb_path).convert("RGB").resize((W, H)))
            d_raw = load_xyz(xyz_path_for(root, sids[i]))
            depth_u8 = np.array(Image.fromarray(
                depth_to_uint8(extract_depth(d_raw), valid_mask_from_xyz(d_raw))).resize((W, H)))
            save_gate_fig(fig_base / cat / f"{tag}_{sids[i].replace('/', '_')}.png",
                          rgb_u8, depth_u8, gts[i], p_rgb, p_dep, fmaps[i],
                          w_img_all[i], w_map_all[i], f"{tag} {sids[i]}")

    tables = REPO_ROOT / "experiments" / "tables"
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables / "E3C_gate_metrics.csv", index=False)

    # Comparison vs all frozen predecessors.
    rgb_df = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    depth_df = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    fixed_df = pd.read_csv(tables / "E3_vs_baselines.csv")
    e3b_df = pd.read_csv(tables / "E3B_vs_all.csv")
    comp = []
    for _, fr in metrics_df.iterrows():
        cat = fr["category"]
        rr = rgb_df[rgb_df.category == cat].iloc[0]
        dr = depth_df[depth_df.category == cat].iloc[0]
        fx = fixed_df[(fixed_df.category == cat)]
        base = {"I-AUROC(max)": (float(rr["image_auroc_max"]), float(dr["image_auroc_max"]), float(fr["image_auroc_max"])),
                "I-AUROC(topk1%)": (float(rr["image_auroc_topk_1pct"]), float(dr["image_auroc_topk_1pct"]), float(fr["image_auroc_topk1"])),
                "P-AUROC": (float(rr["pixel_auroc"]), float(dr["pixel_auroc"]), float(fr["pixel_auroc"])),
                "AUPRO": (float(rr["aupro"]), float(dr["aupro"]), float(fr["aupro"]))}
        for metric, (rv, dv, fv) in base.items():
            fxv = float(fx[fx.metric == metric]["LateFusion"].iloc[0])
            e3b_vals = [float(e3b_df[(e3b_df.category == cat) & (e3b_df.rule == rl) & (e3b_df.metric == metric)]["Rule"].iloc[0])
                        for rl in ("Static", "Validity", "Evidence")]
            comp.append({"category": cat, "metric": metric,
                         "RGB": round(rv, 4), "Depth": round(dv, 4),
                         "FixedMean": round(fxv, 4),
                         "BestE3B": round(max(e3b_vals), 4),
                         "E3C": round(fv, 4),
                         "d_vs_BestSingle": round(fv - max(rv, dv), 4),
                         "d_vs_FixedMean": round(fv - fxv, 4),
                         "d_vs_BestE3B": round(fv - max(e3b_vals), 4)})
    for metric in ("I-AUROC(max)", "I-AUROC(topk1%)", "P-AUROC", "AUPRO"):
        key = {"I-AUROC(max)": "image_auroc_max", "I-AUROC(topk1%)": "image_auroc_topk1",
               "P-AUROC": "pixel_auroc", "AUPRO": "aupro"}[metric]
        fv = float(np.nanmean(metrics_df[key].to_numpy(float)))
        rv = float(np.nanmean(rgb_df[rgb_df.category != "MEAN"][key.replace("image_auroc_topk1", "image_auroc_topk_1pct")].to_numpy(float)))
        dv = float(np.nanmean(depth_df[depth_df.category != "MEAN"][key.replace("image_auroc_topk1", "image_auroc_topk_1pct")].to_numpy(float)))
        fxv = float(np.nanmean(fixed_df[(fixed_df.category != "MEAN") & (fixed_df.metric == metric)]["LateFusion"].to_numpy(float)))
        eb = float(np.nanmean(e3b_df[(e3b_df.category != "MEAN") & (e3b_df.metric == metric)].groupby("category")["Rule"].max().to_numpy(float)))
        comp.append({"category": "MEAN", "metric": metric, "RGB": round(rv, 4), "Depth": round(dv, 4),
                     "FixedMean": round(fxv, 4), "BestE3B": round(eb, 4), "E3C": round(fv, 4),
                     "d_vs_BestSingle": round(fv - max(rv, dv), 4),
                     "d_vs_FixedMean": round(fv - fxv, 4), "d_vs_BestE3B": round(fv - eb, 4)})
    pd.DataFrame(comp).to_csv(tables / "E3C_vs_all.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(tables / "E3C_defect_level.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(tables / "E3C_weights.csv", index=False)

    # Fixed/broken counts vs E3 fixed-mean (validation-p95 operating point).
    e3_pred_base = REPO_ROOT / "experiments" / "predictions" / "E3_late_fusion"
    fb_rows = []
    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib_all = json.load(f)
    for cat in cats:
        thr_c = calib_all["categories"][cat]["score_thresholds"]["fused_max_p95"]
        n_fixed = n_broken = n_both_ok = n_both_bad = 0
        for f in sorted((out_pred / cat).glob("*.npz")):
            z = np.load(f)
            sid = str(z["sample_id"])
            lab = bool(int(z["label"]))
            e3c_ok = bool((float(z["score"]) >= thr_c) == lab)
            ze = np.load(e3_pred_base / cat / f.name)
            e3_ok = bool((float(np.max(ze["map_mean"])) >= thr_c) == lab)
            if e3c_ok and not e3_ok:
                n_fixed += 1
            elif e3_ok and not e3c_ok:
                n_broken += 1
            elif e3c_ok and e3_ok:
                n_both_ok += 1
            else:
                n_both_bad += 1
        fb_rows.append({"category": cat, "fixed": n_fixed, "broken": n_broken,
                        "both_ok": n_both_ok, "both_bad": n_both_bad})
    pd.DataFrame(fb_rows).to_csv(tables / "E3C_fixed_broken.csv", index=False)

    # CPU overhead: gate forward timing (batch 256, CPU).
    gate.eval()
    x = torch.randn(256, len(mu))
    for _ in range(10):
        gate(x)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        gate(x)
        ts.append((time.perf_counter() - t0) * 1000 / 256)
    import os

    print(f"gate overhead: {np.mean(ts):.4f} ms/sample (batch256 CPU)")
    print(f"gate ckpt KB: {os.path.getsize(REPO_ROOT / args.gate_dir / 'gate_best.pt') / 1024:.1f}")
    print("wrote E3C tables + figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
