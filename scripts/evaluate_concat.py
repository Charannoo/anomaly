"""Evaluate E4 concat fusion (Phase 4A) on frozen test predictions protocol.

Per category: rebuild frozen Phase 2 branches + trained fusion, refit the
DECLARED E3 calibration method (validation-normal median/MAD) on E4's own
validation maps, run test inference, fuse calibrated 50/50 mean (invalid ->
RGB only) + max secondary, compute verified metrics, and quantify how much
E4 changed each modality map vs Phase 2.

Writes predictions/E4_concat_fusion/, tables/E4_{metrics,vs_all,map_change,
defect_level,calibration}. No retraining, no test tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.data.mvtec3d import MVTec3DDataset, discover_categories, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.concat import E4ConcatModel
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config


def build_e4(cfg: dict, rgb_ckpt: Path, depth_ckpt: Path, fusion_ckpt: Path | None,
             device: torch.device) -> E4ConcatModel:
    mc = cfg.get("model", {})

    def branch(modality: str, ckpt: Path):
        m = TeacherStudentAD(modality=modality,
                             level_weights=tuple(mc.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
                             pretrained=False, score_mode="max", topk_frac=0.01,
                             smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
        load_checkpoint(ckpt, m)
        return m.eval()

    e4 = E4ConcatModel(branch("rgb", rgb_ckpt), branch("depth", depth_ckpt),
                       proj_dim=int(mc.get("projection_dim", 32)),
                       gamma_init=float(mc.get("gamma_init", 0.0)),
                       smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
    if fusion_ckpt is not None:
        fz = torch.load(str(fusion_ckpt), map_location="cpu", weights_only=False)
        e4.fusion.load_state_dict(fz["fusion_state"])
    return e4.to(device).eval()


def loader_for(cfg: dict, root: Path, category: str, split: str, depth_stats: dict,
               train_aug: bool):
    data_cfg = cfg.get("data", {})
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats,
        depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=train_aug)
    recs = [r for r in discover_split(root, category, split) if r.label == 0] if split != "test" \
        else discover_split(root, category, "test")
    ds = MVTec3DDataset(recs, input_size=input_size, transform=tf, depth_stats=depth_stats)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0), recs


def robust_params(values: np.ndarray):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sc = mad * 1.4826
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = float(np.std(v))
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = 1.0
    return med, float(sc)


@torch.no_grad()
def collect_maps(e4: E4ConcatModel, loader, device):
    rgb_maps, depth_maps, valids, gts = [], [], [], []
    for batch in tqdm(loader, leave=False):
        out = e4.predict(batch["rgb"].to(device), batch["depth"].to(device),
                         batch["valid"].to(device))
        for i in range(batch["rgb"].shape[0]):
            rgb_maps.append(out["rgb_map"][i, 0].cpu().numpy().astype(np.float64))
            depth_maps.append(out["depth_map"][i, 0].cpu().numpy().astype(np.float64))
            valids.append(batch["valid"][i, 0].numpy().astype(bool))
            gts.append(batch["gt"][i].numpy().astype(bool))
    return rgb_maps, depth_maps, valids, gts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--fusion-checkpoint", default="checkpoints/E4_concat_fusion")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    cats = args.categories or cfg.get("data", {}).get("categories") or discover_categories(str(root))
    dev = args.device or cfg.get("device", "auto")
    device = torch.device("cuda" if (dev == "cuda" or (dev == "auto" and torch.cuda.is_available())) else "cpu")

    calib_all: dict = {"method": "robust_median_mad (E3-declared, refit on E4 validation maps)",
                       "fit_on": "validation_normal_only", "categories": {}}
    metric_rows, change_rows, defect_rows, gamma_rows = [], [], [], []
    out_pred = REPO_ROOT / "experiments" / "predictions" / "E4_concat_fusion"
    for cat in cats:
        print(f"== {cat} ==", flush=True)
        r_ckpt = sorted(Path(args.rgb_checkpoint).glob(f"{cat}_best.pt"))[0]
        d_ckpt = sorted(Path(args.depth_checkpoint).glob(f"{cat}_best.pt"))[0]
        f_ckpts = sorted(Path(args.fusion_checkpoint).glob(f"{cat}_best.pt"))
        if not f_ckpts:
            raise FileNotFoundError(f"No E4 checkpoint for {cat}")
        e4 = build_e4(cfg, r_ckpt, d_ckpt, f_ckpts[0], device)
        fz = torch.load(str(f_ckpts[0]), map_location="cpu", weights_only=False)
        gamma_rows.append({"category": cat, "gamma_rgb": round(float(fz.get("gamma_rgb", 0.0)), 5),
                           "gamma_depth": round(float(fz.get("gamma_depth", 0.0)), 5),
                           "val_loss": round(float(fz.get("val_loss", float("nan"))), 5),
                           "fusion_params": sum(p.numel() for p in e4.fusion.parameters())})
        depth_stats = fit_train_depth_stats(root, cat, cfg.get("data", {}).get("depth_norm", "robust_median"),
                                            verbose=False)
        val_loader, _ = loader_for(cfg, root, cat, "validation", depth_stats, False)
        v_r, v_d, v_v, _ = collect_maps(e4, val_loader, device)
        med_r, sc_r = robust_params(np.concatenate([m.ravel() for m in v_r]))
        pooled_d = np.concatenate([m[v].ravel() for m, v in zip(v_d, v_v) if v.any()])
        med_d, sc_d = robust_params(pooled_d)
        calib_all["categories"][cat] = {"rgb": {"median": med_r, "scale": sc_r},
                                        "depth": {"median": med_d, "scale": sc_d}}
        test_loader, recs = loader_for(cfg, root, cat, "test", depth_stats, False)
        t_r, t_d, t_v, gts = collect_maps(e4, test_loader, device)
        labels, defects, ids = [], [], []
        for r in recs:
            labels.append(r.label)
            defects.append(r.defect)
            ids.append(r.sample_id)
        labels_np = np.array(labels)
        fused_mean, fused_max = [], []
        for a, b, v in zip(t_r, t_d, t_v):
            zr = calibrate(a, med_r, sc_r)
            zd = calibrate(b, med_d, sc_d)
            fused_mean.append(np.where(v, 0.5 * zr + 0.5 * zd, zr).astype(np.float32))
            fused_max.append(np.where(v, np.maximum(zr, zd), zr).astype(np.float32))
        (out_pred / cat).mkdir(parents=True, exist_ok=True)
        for sid, a, b, fm, fx, gt, lab, dft in zip(ids, t_r, t_d, fused_mean, fused_max, gts, labels, defects):
            np.savez_compressed(out_pred / cat / f"{sid.replace('/', '_')}.npz",
                                rgb_map=np.asarray(a, dtype=np.float32),
                                depth_map=np.asarray(b, dtype=np.float32),
                                map_mean=fm, map_max=fx, gt=gt, label=np.int64(lab),
                                defect=dft, sample_id=sid)
        sc = {n: np.array([variant_scores(m, None)[n] for m in fused_mean]) for n, _ in SCORE_VARIANTS}
        row = {"category": cat,
               "image_auroc_max": image_auroc(labels_np, sc["max"]),
               "image_auprc_max": image_auprc(labels_np, sc["max"]),
               "image_auroc_topk1": image_auroc(labels_np, sc["topk_1pct"]),
               "pixel_auroc": pixel_auroc(gts, fused_mean),
               "pixel_auprc": pixel_auprc(gts, fused_mean),
               "aupro": aupro(gts, fused_mean),
               "max_image_auroc": image_auroc(labels_np, np.array([variant_scores(m, None)["max"] for m in fused_max])),
               "max_pixel_auroc": pixel_auroc(gts, fused_max),
               "max_aupro": aupro(gts, fused_max),
               "num_test_images": len(ids), "num_anomalous": int(labels_np.sum())}
        metric_rows.append(row)
        # Map-change analysis vs frozen Phase 2 predictions.
        p2_rgb = sorted((REPO_ROOT / "experiments" / "predictions" / "E1_baseline_rgb" / cat).glob("*.npz"))
        p2_depth = sorted((REPO_ROOT / "experiments" / "predictions" / "E2_baseline_depth" / cat).glob("*.npz"))
        p2r = {f.stem: np.load(f)["map"].astype(np.float64) for f in p2_rgb}
        p2d = {f.stem: np.load(f)["map"].astype(np.float64) for f in p2_depth}
        dr, dd, cr, cd = [], [], [], []
        for sid, a, b in zip(ids, t_r, t_d):
            key = sid.replace("/", "_")
            o_r, o_d = p2r[key], p2d[key]
            dr.append(float(np.abs(a - o_r).mean()))
            dd.append(float(np.abs(b - o_d).mean()))
            cr.append(float(np.corrcoef(a.ravel(), o_r.ravel())[0, 1]) if np.std(a) > 0 and np.std(o_r) > 0 else 1.0)
            cd.append(float(np.corrcoef(b.ravel(), o_d.ravel())[0, 1]) if np.std(b) > 0 and np.std(o_d) > 0 else 1.0)
        change_rows.append({"category": cat,
                            "D_rgb_mean": round(float(np.mean(dr)), 5),
                            "D_depth_mean": round(float(np.mean(dd)), 5),
                            "corr_rgb_mean": round(float(np.mean(cr)), 4),
                            "corr_depth_mean": round(float(np.mean(cd)), 4)})
        # Named defect cases.
        normal_idx = [i for i, l in enumerate(labels) if l == 0]
        by_defect: dict[str, list[int]] = {}
        for i, s in enumerate(ids):
            if labels[i] == 1:
                by_defect.setdefault(defects[i], []).append(i)
        for dft, idxs in sorted(by_defect.items()):
            y = np.array([0] * len(normal_idx) + [1] * len(idxs))
            from sklearn.metrics import roc_auc_score as ras
            defect_rows.append({"category": cat, "defect": dft, "n_normal": len(normal_idx),
                                "n_defect": len(idxs),
                                "e4_auroc": round(float(ras(y, np.concatenate([sc["max"][normal_idx], sc["max"][idxs]]))), 4)})

    tables = REPO_ROOT / "experiments" / "tables"
    with open(tables / "E4_concat_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_all, f, indent=2)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables / "E4_concat_metrics.csv", index=False)
    pd.DataFrame(change_rows).to_csv(tables / "E4_map_change.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(tables / "E4_defect_level.csv", index=False)
    pd.DataFrame(gamma_rows).to_csv(tables / "E4_gamma.csv", index=False)

    # Comparison vs all frozen predecessors.
    rgb_df = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    depth_df = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    fixed_df = pd.read_csv(tables / "E3_vs_baselines.csv")
    e3b_df = pd.read_csv(tables / "E3B_vs_all.csv")
    e3c_df = pd.read_csv(tables / "E3C_vs_all.csv")
    comp = []
    for _, fr in metrics_df.iterrows():
        cat = fr["category"]
        rr = rgb_df[rgb_df.category == cat].iloc[0]
        dr = depth_df[depth_df.category == cat].iloc[0]
        base = {"I-AUROC(max)": (float(rr["image_auroc_max"]), float(dr["image_auroc_max"]), float(fr["image_auroc_max"])),
                "I-AUROC(topk1%)": (float(rr["image_auroc_topk_1pct"]), float(dr["image_auroc_topk_1pct"]), float(fr["image_auroc_topk1"])),
                "P-AUROC": (float(rr["pixel_auroc"]), float(dr["pixel_auroc"]), float(fr["pixel_auroc"])),
                "AUPRO": (float(rr["aupro"]), float(dr["aupro"]), float(fr["aupro"]))}
        for metric, (rv, dv, fv) in base.items():
            fxv = float(fixed_df[(fixed_df.category == cat) & (fixed_df.metric == metric)]["LateFusion"].iloc[0])
            e3b_vals = [float(e3b_df[(e3b_df.category == cat) & (e3b_df.rule == rl) & (e3b_df.metric == metric)]["Rule"].iloc[0])
                        for rl in ("Static", "Validity", "Evidence")]
            e3c = float(e3c_df[(e3c_df.category == cat) & (e3c_df.metric == metric)]["E3C"].iloc[0])
            comp.append({"category": cat, "metric": metric, "RGB": round(rv, 4), "Depth": round(dv, 4),
                         "FixedMean": round(fxv, 4), "BestE3B": round(max(e3b_vals), 4),
                         "E3C": round(e3c, 4), "E4": round(fv, 4),
                         "d_vs_RGB": round(fv - rv, 4), "d_vs_BestSingle": round(fv - max(rv, dv), 4),
                         "d_vs_Fixed": round(fv - fxv, 4), "d_vs_BestE3B": round(fv - max(e3b_vals), 4),
                         "d_vs_E3C": round(fv - e3c, 4)})
    for metric in ("I-AUROC(max)", "I-AUROC(topk1%)", "P-AUROC", "AUPRO"):
        key = {"I-AUROC(max)": "image_auroc_max", "I-AUROC(topk1%)": "image_auroc_topk1",
               "P-AUROC": "pixel_auroc", "AUPRO": "aupro"}[metric]
        rkey = {"I-AUROC(max)": "image_auroc_max", "I-AUROC(topk1%)": "image_auroc_topk_1pct",
                "P-AUROC": "pixel_auroc", "AUPRO": "aupro"}[metric]
        fv = float(np.nanmean(metrics_df[key].to_numpy(float)))
        rv = float(np.nanmean(rgb_df[rgb_df.category != "MEAN"][rkey].to_numpy(float)))
        dv = float(np.nanmean(depth_df[depth_df.category != "MEAN"][rkey].to_numpy(float)))
        fxv = float(np.nanmean(fixed_df[(fixed_df.category != "MEAN") & (fixed_df.metric == metric)]["LateFusion"].to_numpy(float)))
        eb = float(np.nanmean(e3b_df[(e3b_df.category != "MEAN") & (e3b_df.metric == metric)].groupby("category")["Rule"].max().to_numpy(float)))
        ec = float(np.nanmean(e3c_df[(e3c_df.category != "MEAN") & (e3c_df.metric == metric)]["E3C"].to_numpy(float)))
        comp.append({"category": "MEAN", "metric": metric, "RGB": round(rv, 4), "Depth": round(dv, 4),
                     "FixedMean": round(fxv, 4), "BestE3B": round(eb, 4), "E3C": round(ec, 4),
                     "E4": round(fv, 4), "d_vs_RGB": round(fv - rv, 4),
                     "d_vs_BestSingle": round(fv - max(rv, dv), 4), "d_vs_Fixed": round(fv - fxv, 4),
                     "d_vs_BestE3B": round(fv - eb, 4), "d_vs_E3C": round(fv - ec, 4)})
    pd.DataFrame(comp).to_csv(tables / "E4_vs_all.csv", index=False)

    # Dev CPU overhead: E4 predict vs summed branch predicts (batch1, CPU).
    e4m = build_e4(load_config(args.config), sorted(Path(args.rgb_checkpoint).glob("bagel_best.pt"))[0],
                   sorted(Path(args.depth_checkpoint).glob("bagel_best.pt"))[0],
                   sorted(Path(args.fusion_checkpoint).glob("bagel_best.pt"))[0], torch.device("cpu"))
    e4m.eval()
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 1, 256, 256)
    v = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    for _ in range(10):
        e4m.predict(x1, x2, v)
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        e4m.predict(x1, x2, v)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"E4 dev latency (bagel, batch1 CPU): mean={np.mean(ts):.1f} ms p50={np.median(ts):.1f} ms "
          f"(Phase 2 single-branch ~17-25 ms; E4 runs both branches + fusion)")
    print("wrote E4 tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
