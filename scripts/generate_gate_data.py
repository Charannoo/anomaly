"""Generate E3C gate supervision from TRAIN-NORMAL samples only (Phase 3C).

For one category: load gate_train/gate_val source images (split manifest),
synthesize CLEAN / RGB_ONLY / DEPTH_ONLY / JOINT variants at 3 severities,
run the FROZEN Phase 2 detectors, and store gate features + branch-quality
targets + validation-normal score distributions.

Writes experiments/gate_data/<category>.npz and <category>_meta.json.
Deterministic: seeds derive from md5(source|kind|subtype|severity).

Usage:
    python scripts/generate_gate_data.py --categories bagel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from xmvad.data import synthetic_anomalies as SA
from xmvad.data.depth import extract_depth, load_xyz, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_split, load_rgb
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.fusion.gate_features import FEATURE_NAMES, gate_features
from xmvad.fusion.score_fusion import EPS, calibrate
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config

SEVERITIES = ("mild", "medium", "strong")
# (kind, subtype, severity) -> kwargs for the perturbation call.
PLAN: list[tuple[str, str, str, dict]] = (
    [("clean", "none", "na", {})]
    + [("rgb_only", "color_patch", s, {"strength": v}) for s, v in
       (("mild", 0.3), ("medium", 0.6), ("strong", 1.0))]
    + [("rgb_only", "scratch", s, {"width": v}) for s, v in
       (("mild", 1), ("medium", 2), ("strong", 3))]
    + [("rgb_only", "stain", s, {"strength": v}) for s, v in
       (("mild", 0.25), ("medium", 0.5), ("strong", 0.8))]
    + [("depth_only", "dent", s, {"amplitude": v}) for s, v in
       (("mild", 0.01), ("medium", 0.03), ("strong", 0.06))]
    + [("depth_only", "bump", s, {"amplitude": v}) for s, v in
       (("mild", 0.01), ("medium", 0.03), ("strong", 0.06))]
    + [("depth_only", "hole", s, {"size_scale": v}) for s, v in
       (("mild", 0.7), ("medium", 1.0), ("strong", 1.5))]
    + [("joint", "joint_blob", s, {"strength": a, "amplitude": b}) for s, a, b in
       (("mild", 0.3, 0.02), ("medium", 0.6, 0.05), ("strong", 1.0, 0.08))]
    + [("joint", "joint_scratch_dent", s, {"width": a, "amplitude": b}) for s, a, b in
       (("mild", 1, 0.02), ("medium", 2, 0.05), ("strong", 3, 0.08))]
)


def det_seed(source_id: str, kind: str, subtype: str, sev: str) -> int:
    h = hashlib.md5(f"{source_id}|{kind}|{subtype}|{sev}".encode()).hexdigest()
    return int(h[:8], 16)


def apply_variant(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                  kind: str, subtype: str, seed: int, kw: dict):
    """Apply perturbation at full-res raw level. Returns (rgb, depth, valid, mask)."""
    if kind == "clean":
        return rgb, depth, valid, np.zeros(rgb.shape[:2], dtype=bool)
    if kind == "rgb_only":
        fn = {"color_patch": SA.perturb_rgb_color_patch, "scratch": SA.perturb_rgb_scratch,
              "stain": SA.perturb_rgb_stain}[subtype]
        rgb_p, mask = fn(rgb, seed=seed, **kw)
        return rgb_p, depth, valid, mask
    if kind == "depth_only":
        if subtype == "hole":
            valid_p, mask = SA.perturb_depth_hole(valid, seed=seed, **kw)
            return rgb, depth, valid_p, mask
        fn = {"dent": SA.perturb_depth_dent, "bump": SA.perturb_depth_bump}[subtype]
        depth_p, mask = fn(depth, valid, seed=seed, **kw)
        return rgb, depth_p, valid, mask
    if kind == "joint":
        fn = {"joint_blob": SA.perturb_joint_blob,
              "joint_scratch_dent": SA.perturb_joint_scratch_dent}[subtype]
        rgb_p, depth_p, mask = fn(rgb, depth, valid, seed=seed, **kw)
        return rgb_p, depth_p, valid, mask
    raise ValueError(kind)


def localization_targets(amap_r: np.ndarray, amap_d: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Pixel-AUPRC per branch vs synthetic GT -> normalized weights."""
    from sklearn.metrics import average_precision_score

    if not np.asarray(gt, dtype=bool).any():
        return 0.5, 0.5  # CLEAN prior: neutral
    y = np.asarray(gt, dtype=bool).ravel()
    q = []
    for m in (amap_r, amap_d):
        s = np.asarray(m, dtype=float).ravel()
        q.append(float(average_precision_score(y, s)) if np.isfinite(s).all() else 0.0)
    s = q[0] + q[1] + EPS
    return q[0] / s, q[1] / s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--rgb-config", default="configs/baseline_rgb.yaml")
    ap.add_argument("--depth-config", default="configs/baseline_depth.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--calibration", default="experiments/calibration/E3_late_fusion_calibration.json")
    ap.add_argument("--split-manifest", default="experiments/calibration/E3C_split_manifest.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rgb_cfg = load_config(args.rgb_config)
    depth_cfg = load_config(args.depth_config)
    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    with open(REPO_ROOT / args.split_manifest, encoding="utf-8") as f:
        manifest = json.load(f)["categories"]
    device = torch.device(args.device)
    out_dir = REPO_ROOT / "experiments" / "gate_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        print(f"== {cat} ==", flush=True)
        data_cfg = rgb_cfg.get("data", {})
        root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
        input_size = tuple(data_cfg.get("input_size", [256, 256]))
        H, W = input_size
        rgb_mean = tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406]))
        rgb_std = tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225]))
        clip = tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None
        depth_stats = fit_train_depth_stats(root, cat, data_cfg.get("depth_norm", "robust_median"),
                                            verbose=False)
        tf = SynchronizedTransform(input_size=input_size, rgb_mean=rgb_mean, rgb_std=rgb_std,
                                   depth_stats=None, depth_clip=clip, train=False)
        # NOTE: depth_stats=None here: raw Z is normalized manually below so
        # synthetic depth edits (in raw units) stay exact; stats only used
        # for the explicit normalize step. See tensorize().
        from xmvad.data.depth import normalize_depth
        models = {}
        for mod, cfg, ckdir in (("rgb", rgb_cfg, args.rgb_checkpoint),
                                ("depth", depth_cfg, args.depth_checkpoint)):
            mc = cfg.get("model", {})
            ckpt = sorted(Path(ckdir).glob(f"{cat}_best.pt"))[0]
            m = TeacherStudentAD(modality=mod,
                                 level_weights=tuple(mc.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
                                 pretrained=False, score_mode="max",
                                 topk_frac=0.01, smooth_sigma=0.0)
            from xmvad.training.checkpoints import load_checkpoint as lc
            lc(ckpt, m)
            models[mod] = m.to(device).eval()
        cm = calib["categories"][cat]

        train_recs = {r.xyz_path.stem: r for r in discover_split(root, cat, "train") if r.label == 0}
        split_of = {}
        for s in manifest[cat]["gate_train"]:
            split_of[s] = "gate_train"
        for s in manifest[cat]["gate_val"]:
            split_of[s] = "gate_val"

        # Validation-normal score distributions (for detection evidence).
        val_max = {"rgb": [], "depth": []}
        for r in [x for x in discover_split(root, cat, "validation") if x.label == 0]:
            xyz = load_xyz(r.xyz_path)
            d0, v0 = extract_depth(xyz), valid_mask_from_xyz(xyz)
            rgb_t, depth_t, valid_t = tensorize(load_rgb(r.rgb_path), d0, v0, tf, depth_stats, clip, H, W)
            with torch.no_grad():
                for mod, ten, vv in (("rgb", rgb_t, None), ("depth", depth_t, valid_t)):
                    am = models[mod].predict(ten.to(device), vv.to(device) if vv is not None else None)["anomaly_map"]
                    m64 = am[0, 0].cpu().numpy().astype(np.float64)
                    ev = m64 if mod == "rgb" else m64[vv[0, 0].numpy()]
                    val_max[mod].append(float(ev.max()) if ev.size else 0.0)
        val_dist = {m: {"median": float(np.median(v)), "p99": float(np.percentile(v, 99))}
                    for m, v in val_max.items()}

        feats, w_img_t, w_map_t, groups, sevs, sids, splits = [], [], [], [], [], [], []
        src_ids = sorted(split_of)
        for stem in tqdm(src_ids, desc=f"gen {cat}", leave=False):
            rec = train_recs[stem]
            rgb0 = np.array(load_rgb(rec.rgb_path))
            xyz0 = load_xyz(rec.xyz_path)
            depth0, valid0 = extract_depth(xyz0), valid_mask_from_xyz(xyz0)
            for kind, subtype, sev, kw in PLAN:
                seed = det_seed(f"{cat}/train/good/{stem}", kind, subtype, sev)
                rgb_p, depth_p, valid_p, mask = apply_variant(rgb0, depth0, valid0, kind, subtype, seed, kw)
                rgb_t, depth_t, valid_t = tensorize(Image.fromarray(rgb_p), depth_p, valid_p, tf,
                                                   depth_stats, clip, H, W)
                gt_small = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
                with torch.no_grad():
                    am_r = models["rgb"].predict(rgb_t.to(device), None)["anomaly_map"][0, 0].cpu().numpy()
                    am_d = models["depth"].predict(depth_t.to(device), valid_t.to(device))["anomaly_map"][0, 0].cpu().numpy()
                zr = calibrate(am_r, cm["rgb"]["median"], cm["rgb"]["scale"])
                zd = calibrate(am_d, cm["depth"]["median"], cm["depth"]["scale"])
                v = valid_t[0, 0].numpy().astype(bool)
                feats.append(gate_features(zr, zd, v))
                # Localization targets from synthetic GT.
                qr, qd = localization_targets(am_r, am_d, gt_small)
                w_map_t.append((qr, qd))
                # Detection targets from validation-normal score distributions.
                sr = float(np.max(am_r[np.isfinite(am_r)]))
                evd = am_d[v] if v.any() else np.array([0.0])
                sd = float(np.max(evd[np.isfinite(evd)])) if evd.size else 0.0
                if kind == "clean":
                    w_img_t.append((0.5, 0.5))
                else:
                    er = max(0.0, (sr - val_dist["rgb"]["median"]) / (val_dist["rgb"]["p99"] - val_dist["rgb"]["median"] + EPS))
                    ed = max(0.0, (sd - val_dist["depth"]["median"]) / (val_dist["depth"]["p99"] - val_dist["depth"]["median"] + EPS))
                    s = er + ed + EPS
                    w_img_t.append((er / s, ed / s))
                groups.append(kind)
                sevs.append(sev)
                sids.append(f"{cat}/train/good/{stem}|{kind}|{subtype}|{sev}")
                splits.append(split_of[stem])
        arr = {
            "features": np.array(feats, dtype=np.float32),
            "w_img_target": np.array(w_img_t, dtype=np.float32),
            "w_map_target": np.array(w_map_t, dtype=np.float32),
            "val_score_dist": np.array([val_dist["rgb"]["median"], val_dist["rgb"]["p99"],
                                        val_dist["depth"]["median"], val_dist["depth"]["p99"]], dtype=np.float64),
        }
        np.savez_compressed(out_dir / f"{cat}.npz", **arr)
        meta = {"category": cat, "feature_names": FEATURE_NAMES,
                "groups": groups, "severities": sevs, "source_ids": sids, "splits": splits,
                "plan": [(k, st, sv, kw) for k, st, sv, kw in PLAN],
                "val_score_dist": val_dist}
        with open(out_dir / f"{cat}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {cat}: {len(feats)} samples", flush=True)
    return 0


def tensorize(rgb_pil, depth_np, valid_np, tf, depth_stats, clip, H, W):
    """Shared eval tensorization: RGB via transform pieces, depth normalized
    explicitly with train stats (keeps synthetic raw-unit edits exact)."""
    from xmvad.data.depth import normalize_depth

    r = rgb_pil.convert("RGB").resize((W, H), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.array(r, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    rgb_t = ((rgb_t - torch.tensor(tf.rgb_mean).view(3, 1, 1)) / torch.tensor(tf.rgb_std).view(3, 1, 1)).unsqueeze(0)
    d = np.array(Image.fromarray(np.asarray(depth_np, dtype=np.float32), mode="F").resize((W, H), Image.NEAREST), dtype=np.float32)
    v = np.array(Image.fromarray(valid_np.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    dn = normalize_depth(d, depth_stats, valid_mask=v, clip=clip)
    depth_t = torch.from_numpy(dn).unsqueeze(0).unsqueeze(0)
    valid_t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
    return rgb_t.float(), depth_t.float(), valid_t.bool()


if __name__ == "__main__":
    raise SystemExit(main())
