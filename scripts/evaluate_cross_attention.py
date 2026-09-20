"""Evaluate E5/E6/E7 XMV cross-attention (Phase 4B) on the frozen protocol.

For each experiment and category: rebuild frozen Phase 2 branches + trained
XMV attention, refit the DECLARED E3 calibration (validation-normal
median/MAD) on that experiment's own validation maps, run test inference,
fuse calibrated 50/50 mean (invalid -> RGB only), compute verified metrics
(max primary + top 0.1%/1% secondary), save per-image predictions, measure
map change vs Phase 2 and vs E4, and log cross-modal interaction diagnostics
(gammas, residual magnitudes, attention entropy, valid-depth attention mass).

Then the combined Phase 4B analysis: E7 vs E4 / fixed / best-single table,
paired image-level bootstrap (E7-E4), directional E5/E6/E7 comparison, saved
"sanity" cross-modal interaction maps for selected samples (labeled
INTERACTION, never explanation), and a development CPU overhead benchmark of
E4/E5/E6/E7. No retraining, no test tuning.
"""

from __future__ import annotations

import argparse
import json
import os
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
from xmvad.models.fusion.cross_attention import NUM_HEADS, XMVModel
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config

EXPERIMENTS = [
    ("E5_cross_attn_rgb", "configs/cross_attention_rgb_from_depth.yaml"),
    ("E6_cross_attn_depth", "configs/cross_attention_depth_from_rgb.yaml"),
    ("E7_xmv_bidirectional", "configs/xmv_ad.yaml"),
]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def build_xmv(cfg: dict, rgb_ckpt: Path, depth_ckpt: Path, attn_ckpt: Path | None,
              device: torch.device) -> XMVModel:
    mc = cfg.get("model", {})

    def branch(modality: str, ckpt: Path):
        m = TeacherStudentAD(modality=modality,
                             level_weights=tuple(mc.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
                             pretrained=False, score_mode="max", topk_frac=0.01,
                             smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
        load_checkpoint(ckpt, m)
        return m.eval()

    m = XMVModel(branch("rgb", rgb_ckpt), branch("depth", depth_ckpt),
                 mode=str(mc.get("mode", "bidirectional")),
                 attn_dim=int(mc.get("attention_dim", 32)),
                 num_heads=int(mc.get("num_heads", 2)),
                 gamma_init=float(mc.get("gamma_init", 0.0)),
                 smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
    if attn_ckpt is not None:
        ck = torch.load(str(attn_ckpt), map_location="cpu", weights_only=False)
        m.attn.load_state_dict(ck["attn_state"])
    return m.to(device).eval()


def loader_for(cfg: dict, root: Path, category: str, split: str, depth_stats: dict):
    data_cfg = cfg.get("data", {})
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats,
        depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=False)
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


def _attn_stats(w: torch.Tensor, mask_bad_rows: bool) -> tuple[float, float]:
    """Return (entropy_nats, mean_row_mass) for one attention tensor (B,H,N,N).

    Fully-masked rows (all-zero after masking) are excluded when asked.
    """
    w = w.float().numpy()
    mass = w.sum(-1)  # (B,H,N)
    if mask_bad_rows:
        keep = mass > 1e-6
        sel = w[keep]
        if sel.size == 0:
            return float("nan"), 0.0
        ent = float((-(sel * np.log(np.clip(sel, 1e-12, None))).sum(-1)).mean())
        return ent, float(mass[keep].mean())
    ent = float((-(w * np.log(np.clip(w, 1e-12, None))).sum(-1)).mean())
    return ent, float(mass.mean())


@torch.no_grad()
def collect(m, loader, device, save_attn_for: set[str] | None = None,
            attn_dir: Path | None = None):
    """Collect maps, gts, ids, per-image diagnostics, optional attention maps."""
    rgb_maps, depth_maps, valids, gts, ids, labels, defects = [], [], [], [], [], [], []
    diag = {"entropy_rgb": [], "entropy_depth": [], "mass_rgb": [],
            "res_rgb": [], "res_depth": []}
    for batch in tqdm(loader, leave=False):
        out = m.predict(batch["rgb"].to(device), batch["depth"].to(device),
                        batch["valid"].to(device))
        bs = batch["rgb"].shape[0]
        # Residual magnitudes from stashed deltas.
        for tag in ("rgb", "depth"):
            d = m.last_deltas[tag]
            if d is not None:
                diag[f"res_{tag}"].append(float(d.abs().mean().item()))
        # Attention entropy / valid mass.
        if m.attn.last_attn_rgb is not None:
            e, mass = _attn_stats(m.attn.last_attn_rgb, mask_bad_rows=True)
            if np.isfinite(e):
                diag["entropy_rgb"].append(e)
            diag["mass_rgb"].append(mass)
        if m.attn.last_attn_depth is not None:
            e, _ = _attn_stats(m.attn.last_attn_depth, mask_bad_rows=False)
            if np.isfinite(e):
                diag["entropy_depth"].append(e)
        if save_attn_for is not None and attn_dir is not None:
            for i in range(bs):
                sid = batch["sample_id"][i] if "sample_id" in batch else None
                if sid in save_attn_for:
                    _save_interaction(m, i, attn_dir, sid)
        for i in range(bs):
            rgb_maps.append(out["rgb_map"][i, 0].cpu().numpy().astype(np.float64))
            depth_maps.append(out["depth_map"][i, 0].cpu().numpy().astype(np.float64))
            valids.append(batch["valid"][i, 0].numpy().astype(bool))
            gts.append(batch["gt"][i].numpy().astype(bool))
            ids.append(batch["sample_id"][i] if "sample_id" in batch else str(len(ids)))
            labels.append(int(batch["label"][i]))
            defects.append(batch["defect"][i])
    return {"rgb": rgb_maps, "depth": depth_maps, "valid": valids, "gt": gts,
            "ids": ids, "labels": labels, "defects": defects, "diag": diag}


def _save_interaction(m: XMVModel, i: int, attn_dir: Path, sid: str) -> None:
    """Save head-averaged token-level attention as spatial INTERACTION maps."""
    attn_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for tag, w in (("rgb_from_depth", m.attn.last_attn_rgb),
                   ("depth_from_rgb", m.attn.last_attn_depth)):
        if w is None:
            continue
        wi = w[i].mean(0).numpy()  # (Nq,Nk) averaged over heads
        n = int(round(wi.shape[0] ** 0.5))
        if n * n != wi.shape[0]:
            continue
        # Which source tokens are attended to (key importance).
        key_imp = wi.mean(0).reshape(n, n)
        arrays[f"{tag}_key_importance"] = key_imp.astype(np.float32)
    if arrays:
        np.savez_compressed(attn_dir / f"{sid.replace('/', '_')}_interaction.npz", **arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-maps", action="store_true", help="skip interaction map dumps")
    args = ap.parse_args()

    tables = REPO_ROOT / "experiments" / "tables"
    pred_root = REPO_ROOT / "experiments" / "predictions"
    attn_root = REPO_ROOT / "experiments" / "attention"
    all_metrics: dict[str, pd.DataFrame] = {}
    all_gamma: list[dict] = []
    all_diag: list[dict] = []
    all_change: dict[str, list] = {}
    all_defect: list[dict] = []

    for exp, cfg_path in EXPERIMENTS:
        cfg = load_config(cfg_path)
        root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
        cats = args.categories or cfg.get("data", {}).get("categories") or discover_categories(str(root))
        dev = args.device or cfg.get("runtime", {}).get("device", "auto")
        device = torch.device("cuda" if (dev == "cuda" or (dev == "auto" and torch.cuda.is_available())) else "cpu")
        print(f"\n######## {exp} ({cfg_path}) ########", flush=True)
        calib = {"method": "robust_median_mad (E3-declared, refit on this experiment's validation maps)",
                 "fit_on": "validation_normal_only", "categories": {}}
        metric_rows, change_rows = [], []
        for cat in cats:
            print(f"== {exp} :: {cat} ==", flush=True)
            r_ckpt = sorted(Path(args.rgb_checkpoint).glob(f"{cat}_best.pt"))[0]
            d_ckpt = sorted(Path(args.depth_checkpoint).glob(f"{cat}_best.pt"))[0]
            a_ckpts = sorted((REPO_ROOT / "checkpoints" / exp).glob(f"{cat}_best.pt"))
            if not a_ckpts:
                raise FileNotFoundError(f"No {exp} checkpoint for {cat}")
            ck = torch.load(str(a_ckpts[0]), map_location="cpu", weights_only=False)
            m = build_xmv(cfg, r_ckpt, d_ckpt, a_ckpts[0], device)
            all_gamma.append({"experiment": exp, "category": cat,
                              "gamma_rgb_from_depth": round(float(ck.get("gamma_rd", 0.0)), 5),
                              "gamma_depth_from_rgb": round(float(ck.get("gamma_dr", 0.0)), 5),
                              "val_loss": round(float(ck.get("val_loss", float("nan"))), 5),
                              "attn_params": int(ck.get("attn_params", sum(p.numel() for p in m.attn.parameters())))})
            depth_stats = fit_train_depth_stats(root, cat, cfg.get("data", {}).get("depth_norm", "robust_median"),
                                                verbose=False)
            val_loader, _ = loader_for(cfg, root, cat, "validation", depth_stats)
            v = collect(m, val_loader, device)
            med_r, sc_r = robust_params(np.concatenate([x.ravel() for x in v["rgb"]]))
            pooled_d = np.concatenate([x[vv].ravel() for x, vv in zip(v["depth"], v["valid"]) if vv.any()])
            med_d, sc_d = robust_params(pooled_d)
            calib["categories"][cat] = {"rgb": {"median": med_r, "scale": sc_r},
                                        "depth": {"median": med_d, "scale": sc_d}}
            # Selected samples for interaction maps: first normal + first anomalous.
            selected: set[str] = set()
            if not args.no_maps and exp == "E7_xmv_bidirectional":
                test_recs_pre = discover_split(root, cat, "test")
                normals = [r.sample_id for r in test_recs_pre if r.label == 0][:1]
                anoms = [r.sample_id for r in test_recs_pre if r.label == 1][:1]
                selected = set(normals + anoms)
            test_loader, recs = loader_for(cfg, root, cat, "test", depth_stats)
            t = collect(m, test_loader, device, save_attn_for=selected,
                        attn_dir=attn_root / exp / cat)
            labels_np = np.array([r.label for r in recs])
            ids = [r.sample_id for r in recs]
            defects = [r.defect for r in recs]
            fused_mean, fused_max = [], []
            for a, b, vv in zip(t["rgb"], t["depth"], t["valid"]):
                zr = calibrate(a, med_r, sc_r)
                zd = calibrate(b, med_d, sc_d)
                fused_mean.append(np.where(vv, 0.5 * zr + 0.5 * zd, zr).astype(np.float32))
                fused_max.append(np.where(vv, np.maximum(zr, zd), zr).astype(np.float32))
            (pred_root / exp / cat).mkdir(parents=True, exist_ok=True)
            for sid, a, b, fm, fx, gt, lab, dft in zip(ids, t["rgb"], t["depth"], fused_mean,
                                                       fused_max, t["gt"], labels_np, defects):
                np.savez_compressed(pred_root / exp / cat / f"{sid.replace('/', '_')}.npz",
                                    rgb_map=np.asarray(a, dtype=np.float32),
                                    depth_map=np.asarray(b, dtype=np.float32),
                                    map_mean=fm, map_max=fx, gt=gt, label=np.int64(lab),
                                    defect=dft, sample_id=sid)
            sc = {n: np.array([variant_scores(mm, None)[n] for mm in fused_mean]) for n, _ in SCORE_VARIANTS}
            row = {"category": cat,
                   "image_auroc_max": image_auroc(labels_np, sc["max"]),
                   "image_auprc_max": image_auprc(labels_np, sc["max"]),
                   "image_auroc_topk_0.1pct": image_auroc(labels_np, sc["topk_0.1pct"]),
                   "image_auprc_topk_0.1pct": image_auprc(labels_np, sc["topk_0.1pct"]),
                   "image_auroc_topk_1pct": image_auroc(labels_np, sc["topk_1pct"]),
                   "image_auprc_topk_1pct": image_auprc(labels_np, sc["topk_1pct"]),
                   "pixel_auroc": pixel_auroc(t["gt"], fused_mean),
                   "pixel_auprc": pixel_auprc(t["gt"], fused_mean),
                   "aupro": aupro(t["gt"], fused_mean),
                   "num_test_images": len(ids), "num_anomalous": int(labels_np.sum())}
            metric_rows.append(row)
            # Diagnostics averaged over test images.
            dg = t["diag"]
            all_diag.append({"experiment": exp, "category": cat,
                             "res_mag_rgb": round(float(np.mean(dg["res_rgb"])) if dg["res_rgb"] else float("nan"), 6),
                             "res_mag_depth": round(float(np.mean(dg["res_depth"])) if dg["res_depth"] else float("nan"), 6),
                             "attn_entropy_rgb": round(float(np.nanmean(dg["entropy_rgb"])) if dg["entropy_rgb"] else float("nan"), 4),
                             "attn_entropy_depth": round(float(np.nanmean(dg["entropy_depth"])) if dg["entropy_depth"] else float("nan"), 4),
                             "valid_depth_attn_mass": round(float(np.mean(dg["mass_rgb"])) if dg["mass_rgb"] else float("nan"), 4)})
            # Map change vs Phase 2 and vs E4.
            p2r = {f.stem: np.load(f)["map"].astype(np.float64)
                   for f in (pred_root / "E1_baseline_rgb" / cat).glob("*.npz")}
            p2d = {f.stem: np.load(f)["map"].astype(np.float64)
                   for f in (pred_root / "E2_baseline_depth" / cat).glob("*.npz")}
            e4files = {f.stem: np.load(f) for f in (pred_root / "E4_concat_fusion" / cat).glob("*.npz")}
            dr, dd, cr, cd, dr4, dd4, cr4, cd4 = [], [], [], [], [], [], [], []
            for sid, a, b in zip(ids, t["rgb"], t["depth"]):
                key = sid.replace("/", "_")
                o_r, o_d = p2r[key], p2d[key]
                dr.append(float(np.abs(a - o_r).mean()))
                dd.append(float(np.abs(b - o_d).mean()))
                cr.append(float(np.corrcoef(a.ravel(), o_r.ravel())[0, 1]) if np.std(a) > 0 and np.std(o_r) > 0 else 1.0)
                cd.append(float(np.corrcoef(b.ravel(), o_d.ravel())[0, 1]) if np.std(b) > 0 and np.std(o_d) > 0 else 1.0)
                if key in e4files:
                    e4 = e4files[key]
                    er = e4["rgb_map"].astype(np.float64)
                    ed = e4["depth_map"].astype(np.float64)
                    dr4.append(float(np.abs(a - er).mean()))
                    dd4.append(float(np.abs(b - ed).mean()))
                    cr4.append(float(np.corrcoef(a.ravel(), er.ravel())[0, 1]) if np.std(a) > 0 and np.std(er) > 0 else 1.0)
                    cd4.append(float(np.corrcoef(b.ravel(), ed.ravel())[0, 1]) if np.std(b) > 0 and np.std(ed) > 0 else 1.0)
            change_rows.append({"category": cat,
                                "D_rgb_mean": round(float(np.mean(dr)), 5),
                                "D_depth_mean": round(float(np.mean(dd)), 5),
                                "corr_rgb_mean": round(float(np.mean(cr)), 4),
                                "corr_depth_mean": round(float(np.mean(cd)), 4),
                                "D_rgb_vsE4": round(float(np.mean(dr4)), 5) if dr4 else float("nan"),
                                "D_depth_vsE4": round(float(np.mean(dd4)), 5) if dd4 else float("nan"),
                                "corr_rgb_vsE4": round(float(np.mean(cr4)), 4) if cr4 else float("nan"),
                                "corr_depth_vsE4": round(float(np.mean(cd4)), 4) if cd4 else float("nan")})
            # Named defect cases.
            normal_idx = [i for i, l in enumerate(labels_np) if l == 0]
            by_defect: dict[str, list[int]] = {}
            for i, s in enumerate(ids):
                if labels_np[i] == 1:
                    by_defect.setdefault(defects[i], []).append(i)
            from sklearn.metrics import roc_auc_score as ras
            for dft, idxs in sorted(by_defect.items()):
                y = np.array([0] * len(normal_idx) + [1] * len(idxs))
                all_defect.append({"experiment": exp, "category": cat, "defect": dft,
                                   "n_normal": len(normal_idx), "n_defect": len(idxs),
                                   "auroc": round(float(ras(y, np.concatenate([sc["max"][normal_idx], sc["max"][idxs]]))), 4)})
        metrics_df = pd.DataFrame(metric_rows)
        mean_row = {c: (round(float(np.nanmean(metrics_df[c].to_numpy(float))), 4)
                        if c not in ("category", "num_test_images", "num_anomalous")
                        else ("MEAN" if c == "category" else int(metrics_df[c].sum())))
                    for c in metrics_df.columns}
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
        metrics_df.to_csv(tables / f"{exp}_metrics.csv", index=False)
        pd.DataFrame(change_rows).to_csv(tables / f"{exp}_map_change.csv", index=False)
        with open(tables / f"{exp}_calibration.json", "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        all_metrics[exp] = metrics_df

    pd.DataFrame(all_gamma).to_csv(tables / "E5E6E7_gamma.csv", index=False)
    pd.DataFrame(all_diag).to_csv(tables / "E5E6E7_attention_diag.csv", index=False)
    pd.DataFrame(all_defect).to_csv(tables / "E5E6E7_defect_level.csv", index=False)

    # ---- combined table: E7 vs E4 / fixed / best-single, all methods ----
    rgb_df = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    depth_df = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    fixed_df = pd.read_csv(tables / "E3_vs_baselines.csv")
    e4_df = pd.read_csv(tables / "E4_concat_metrics.csv")

    fixed_auprc_df = pd.read_csv(tables / "E3_late_fusion_metrics.csv")

    def _fixed_auprc(df: pd.DataFrame, cat: str, metric: str) -> float:
        col = {"I-AUPRC(max)": "mean_image_auprc_max", "P-AUPRC": "mean_pixel_auprc"}.get(metric)
        if col is None or col not in df.columns:
            return np.nan
        s = df[df.category == cat]
        return float(s[col].iloc[0]) if not s.empty else np.nan

    METRICS = {"I-AUROC(max)": "image_auroc_max", "I-AUROC(topk0.1%)": "image_auroc_topk_0.1pct",
               "I-AUROC(topk1%)": "image_auroc_topk_1pct", "I-AUPRC(max)": "image_auprc_max",
               "P-AUROC": "pixel_auroc", "P-AUPRC": "pixel_auprc", "AUPRO": "aupro"}
    comp_rows = []
    cats_all = [c for c in all_metrics["E7_xmv_bidirectional"]["category"].tolist() if c != "MEAN"]
    for cat in cats_all + ["MEAN"]:
        for metric, col in METRICS.items():
            vals = {}
            for exp in ("E5_cross_attn_rgb", "E6_cross_attn_depth", "E7_xmv_bidirectional"):
                df = all_metrics[exp]
                sub = df[df.category == cat]
                vals[exp] = float(sub[col].iloc[0]) if not sub.empty else np.nan
            rr = float(rgb_df[rgb_df.category == cat][col].iloc[0]) if col in rgb_df.columns else np.nan
            dd = float(depth_df[depth_df.category == cat][col].iloc[0]) if col in depth_df.columns else np.nan
            fx = np.nan
            if cat == "MEAN":
                s = fixed_df[(fixed_df.category != "MEAN") & (fixed_df.metric == metric)]
                if not s.empty:
                    fx = float(np.nanmean(s["LateFusion"].to_numpy(float)))
                else:
                    fx = _fixed_auprc(fixed_auprc_df, cat, metric)
            else:
                s = fixed_df[(fixed_df.category == cat) & (fixed_df.metric == metric)]
                if not s.empty:
                    fx = float(s["LateFusion"].iloc[0])
                else:
                    fx = _fixed_auprc(fixed_auprc_df, cat, metric)
            e4 = np.nan
            e4_col = {"I-AUROC(topk1%)": "image_auroc_topk1"}.get(metric, col)
            if e4_col in e4_df.columns:
                ss = e4_df[e4_df.category == cat]
                if not ss.empty:
                    e4 = float(ss[e4_col].iloc[0])
                elif cat == "MEAN":
                    e4 = float(np.nanmean(e4_df[e4_df.category != "MEAN"][e4_col].to_numpy(float)))
            e7 = vals["E7_xmv_bidirectional"]
            comp_rows.append({"category": cat, "metric": metric,
                              "RGB": round(rr, 4) if np.isfinite(rr) else np.nan,
                              "Depth": round(dd, 4) if np.isfinite(dd) else np.nan,
                              "FixedMean": round(fx, 4) if np.isfinite(fx) else np.nan,
                              "E4": round(e4, 4) if np.isfinite(e4) else np.nan,
                              "E5": round(vals["E5_cross_attn_rgb"], 4),
                              "E6": round(vals["E6_cross_attn_depth"], 4),
                              "E7": round(e7, 4),
                              "d_vs_E4": round(e7 - e4, 4) if np.isfinite(e4) else np.nan,
                              "d_vs_Fixed": round(e7 - fx, 4) if np.isfinite(fx) else np.nan,
                              "d_vs_BestSingle": round(e7 - max(rr, dd), 4) if np.isfinite(max(rr, dd)) else np.nan})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(tables / "E7_vs_all.csv", index=False)

    # ---- directionality: best direction per category/metric ----
    dir_rows = []
    for cat in cats_all:
        for metric in ("I-AUROC(max)", "P-AUROC", "AUPRO"):
            sub = comp[(comp.category == cat) & (comp.metric == metric)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            best = max((row["E5"], "E5"), (row["E6"], "E6"), (row["E7"], "E7"))
            dir_rows.append({"category": cat, "metric": metric,
                             "E5_rgb_from_depth": row["E5"], "E6_depth_from_rgb": row["E6"],
                             "E7_bidirectional": row["E7"],
                             "best_direction": best[1],
                             "bidir_minus_best_oneway": round(float(row["E7"]) - float(best[0]), 4)})
    pd.DataFrame(dir_rows).to_csv(tables / "E5E6E7_directionality.csv", index=False)

    # ---- paired bootstrap: E7 vs E4, image-level AUROC (max readout) ----
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rows = []
    pooled_e4, pooled_e7, pooled_lab = [], [], []
    for cat in cats_all:
        lab, e4s, e7s = [], [], []
        for f in sorted((pred_root / "E7_xmv_bidirectional" / cat).glob("*.npz")):
            z7 = np.load(f)
            key = f.stem
            fe4 = pred_root / "E4_concat_fusion" / cat / f.name
            if not fe4.exists():
                continue
            z4 = np.load(fe4)
            lab.append(int(z7["label"]))
            e4s.append(variant_scores(z4["map_mean"], None)["max"])
            e7s.append(variant_scores(z7["map_mean"], None)["max"])
        lab = np.array(lab); e4s = np.array(e4s); e7s = np.array(e7s)
        if lab.size == 0 or lab.sum() == 0 or lab.sum() == lab.size:
            continue
        from sklearn.metrics import roc_auc_score as ras
        delta = float(ras(lab, e7s) - ras(lab, e4s))
        ds = []
        n = lab.size
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n, size=n)
            if lab[idx].sum() in (0, n):
                continue
            ds.append(ras(lab[idx], e7s[idx]) - ras(lab[idx], e4s[idx]))
        ds = np.array(ds)
        boot_rows.append({"category": cat, "delta_E7_minus_E4": round(delta, 4),
                          "ci_lo": round(float(np.percentile(ds, 2.5)), 4),
                          "ci_hi": round(float(np.percentile(ds, 97.5)), 4),
                          "n_bootstrap": int(ds.size),
                          "excludes_zero": bool(np.percentile(ds, 2.5) > 0 or np.percentile(ds, 97.5) < 0)})
        pooled_e4.extend(e4s.tolist()); pooled_e7.extend(e7s.tolist()); pooled_lab.extend(lab.tolist())
    if pooled_lab:
        from sklearn.metrics import roc_auc_score as ras
        pl = np.array(pooled_lab); pe4 = np.array(pooled_e4); pe7 = np.array(pooled_e7)
        ds = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, pl.size, size=pl.size)
            if pl[idx].sum() in (0, pl.size):
                continue
            ds.append(ras(pl[idx], pe7[idx]) - ras(pl[idx], pe4[idx]))
        ds = np.array(ds)
        boot_rows.append({"category": "POOLED", "delta_E7_minus_E4": round(float(ras(pl, pe7) - ras(pl, pe4)), 4),
                          "ci_lo": round(float(np.percentile(ds, 2.5)), 4),
                          "ci_hi": round(float(np.percentile(ds, 97.5)), 4),
                          "n_bootstrap": int(ds.size),
                          "excludes_zero": bool(np.percentile(ds, 2.5) > 0 or np.percentile(ds, 97.5) < 0)})
    pd.DataFrame(boot_rows).to_csv(tables / "E7_vs_E4_bootstrap.csv", index=False)

    # ---- development CPU overhead benchmark: E4/E5/E6/E7 ----
    bench_rows = []
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 1, 256, 256)
    vv = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    bag_r = sorted(Path(args.rgb_checkpoint).glob("bagel_best.pt"))[0]
    bag_d = sorted(Path(args.depth_checkpoint).glob("bagel_best.pt"))[0]
    f4 = sorted((REPO_ROOT / "checkpoints" / "E4_concat_fusion").glob("bagel_best.pt"))[0]

    def _branch(modality, ckpt):
        mm = TeacherStudentAD(modality=modality, pretrained=False, score_mode="max", topk_frac=0.01)
        load_checkpoint(ckpt, mm)
        return mm.eval()

    e4m = E4ConcatModel(_branch("rgb", bag_r), _branch("depth", bag_d), proj_dim=32).eval()
    e4m.fusion.load_state_dict(torch.load(str(f4), map_location="cpu", weights_only=False)["fusion_state"])
    m5 = build_xmv(load_config("configs/cross_attention_rgb_from_depth.yaml"), bag_r, bag_d,
                   sorted((REPO_ROOT / "checkpoints" / "E5_cross_attn_rgb").glob("bagel_best.pt"))[0], torch.device("cpu"))
    m6 = build_xmv(load_config("configs/cross_attention_depth_from_rgb.yaml"), bag_r, bag_d,
                   sorted((REPO_ROOT / "checkpoints" / "E6_cross_attn_depth").glob("bagel_best.pt"))[0], torch.device("cpu"))
    m7 = build_xmv(load_config("configs/xmv_ad.yaml"), bag_r, bag_d,
                   sorted((REPO_ROOT / "checkpoints" / "E7_xmv_bidirectional").glob("bagel_best.pt"))[0], torch.device("cpu"))
    for name, model, ckpt in (
            ("E4_concat_fusion", e4m, f4),
            ("E5_cross_attn_rgb", m5, sorted((REPO_ROOT / "checkpoints" / "E5_cross_attn_rgb").glob("bagel_best.pt"))[0]),
            ("E6_cross_attn_depth", m6, sorted((REPO_ROOT / "checkpoints" / "E6_cross_attn_depth").glob("bagel_best.pt"))[0]),
            ("E7_xmv_bidirectional", m7, sorted((REPO_ROOT / "checkpoints" / "E7_xmv_bidirectional").glob("bagel_best.pt"))[0])):
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model.predict(x1, x2, vv)
            ts = []
            for _ in range(100):
                t0 = time.perf_counter()
                model.predict(x1, x2, vv)
                ts.append((time.perf_counter() - t0) * 1000)
        params = model.attn.count_parameters() if hasattr(model, "attn") else model.fusion.count_parameters()
        bench_rows.append({"experiment": name,
                           "trainable_params": int(params),
                           "checkpoint_kb": round(os.path.getsize(ckpt) / 1024.0, 1),
                           "predict_mean_ms": round(float(np.mean(ts)), 1),
                           "predict_p50_ms": round(float(np.median(ts)), 1),
                           "device": "cpu", "batch_size": 1, "resolution": 256})
    pd.DataFrame(bench_rows).to_csv(tables / "E5E6E7_dev_latency.csv", index=False)
    print("wrote Phase 4B tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())