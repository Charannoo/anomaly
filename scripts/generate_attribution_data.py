"""Phase 5, todo 7: controlled modality-attribution benchmark (TRAIN-NORMAL).

Deterministically inject RGB-only / depth-only / joint anomalies into
TRAIN-NORMAL samples (never test data) using the pre-registered synthetic
generator (``xmvad.data.synthetic_anomalies``), at severities mild / medium /
strong, run the FROZEN E4 detections, and store branch maps + calibrated
evidence + decision scores. No E4 retraining and no test samples.

For each sample (source, kind, subtype, severity): seed = md5 of that tuple,
same rule as Phase 3C ''generate_gate_data''.

Evidence definition (Phase 5, section 7): region R = injected mask;
E_rgb/E_depth = robust top-k (1%) OR mean of calibrated values inside R;
p_rgb = pos(E_rgb)/(pos(E_rgb)+pos(E_depth)+eps); output classes per
``xmvad.metrics.attribution.evidence_class``.

Writes experiments/attribution/attribution_data/<category>.npz and
experiments/attribution/attribution_meta/<category>.json.
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
from xmvad.data.depth import extract_depth, load_xyz, normalize_depth, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_split, load_rgb
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.attribution import (decision_ablation, evidence_class,
                                       region_evidence)
from xmvad.metrics.scores import variant_scores
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.concat import E4ConcatModel
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config

KINDS = ("rgb_only", "depth_only", "joint")
SEVERITIES = ("mild", "medium", "strong")
PLAN = (
    [("rgb_only", "color_patch", s, {"strength": v}) for s, v in
     (("mild", 0.3), ("medium", 0.6), ("strong", 1.0))]
    + [("depth_only", "dent", s, {"amplitude": v}) for s, v in
       (("mild", 0.01), ("medium", 0.03), ("strong", 0.06))]
    + [("joint", "joint_blob", s, {"strength": a, "amplitude": b}) for s, a, b in
       (("mild", 0.3, 0.02), ("medium", 0.6, 0.05), ("strong", 1.0, 0.08))]
)
DEFAULT_SOURCES = 40  # train-normal samples per category (deterministically picked)


def det_seed(source_id: str, kind: str, subtype: str, sev: str) -> int:
    h = hashlib.md5(f"{source_id}|{kind}|{subtype}|{sev}".encode()).hexdigest()
    return int(h[:8], 16)


def apply_variant(rgb, depth, valid, kind, subtype, seed, kw):
    if kind == "rgb_only":
        rgb_p, mask = SA.perturb_rgb_color_patch(rgb, seed=seed, **kw)
        return rgb_p, depth, valid, mask
    if kind == "depth_only":
        depth_p, mask = SA.perturb_depth_dent(depth, valid, seed=seed, **kw)
        return rgb, depth_p, valid, mask
    if kind == "joint":
        rgb_p, depth_p, mask = SA.perturb_joint_blob(rgb, depth, valid, seed=seed, **kw)
        return rgb_p, depth_p, valid, mask
    raise ValueError(kind)


def build_e4(cfg, cat, rgb_ckpt_dir, depth_ckpt_dir, fusion_ckpt_dir, device):
    mc = cfg.get("model", {})

    def branch(modality, ckdir):
        m = TeacherStudentAD(modality=modality,
                             level_weights=tuple(mc.get("level_weights", [0.5, 0.5])),
                             pretrained=False, score_mode="max", topk_frac=0.01,
                             smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
        load_checkpoint(sorted(Path(ckdir).glob(f"{cat}_best.pt"))[0], m)
        return m.eval()

    e4 = E4ConcatModel(branch("rgb", rgb_ckpt_dir), branch("depth", depth_ckpt_dir),
                       proj_dim=int(mc.get("projection_dim", 32)),
                       gamma_init=float(mc.get("gamma_init", 0.0)),
                       smooth_sigma=float(mc.get("smooth_sigma", 0.0)))
    fz = torch.load(str(sorted(Path(fusion_ckpt_dir).glob(f"{cat}_best.pt"))[0]),
                    map_location="cpu", weights_only=False)
    e4.fusion.load_state_dict(fz["fusion_state"])
    return e4.to(device).eval()


def tensorize(rgb_pil, depth_np, valid_np, tf, depth_stats, clip, H, W):
    r = rgb_pil.convert("RGB").resize((W, H), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.array(r, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    rgb_t = ((rgb_t - torch.tensor(tf.rgb_mean).view(3, 1, 1)) /
             torch.tensor(tf.rgb_std).view(3, 1, 1)).unsqueeze(0)
    d = np.array(Image.fromarray(np.asarray(depth_np, dtype=np.float32), mode="F")
                 .resize((W, H), Image.NEAREST), dtype=np.float32)
    v = np.array(Image.fromarray(valid_np.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    dn = normalize_depth(d, depth_stats, valid_mask=v, clip=clip)
    depth_t = torch.from_numpy(dn).unsqueeze(0).unsqueeze(0)
    valid_t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
    return rgb_t.float(), depth_t.float(), valid_t.bool()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--fusion-checkpoint", default="checkpoints/E4_concat_fusion")
    ap.add_argument("--calibration", default="experiments/tables/E4_concat_calibration.json")
    ap.add_argument("--sources", type=int, default=DEFAULT_SOURCES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    H, W = input_size
    rgb_mean = tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406]))
    rgb_std = tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225]))
    clip = tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None
    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)["categories"]
    device = torch.device(args.device)

    for cat in args.categories:
        print(f"== {cat} ==", flush=True)
        depth_stats = fit_train_depth_stats(root, cat, data_cfg.get("depth_norm", "robust_median"),
                                            verbose=False)
        tf = SynchronizedTransform(input_size=input_size, rgb_mean=rgb_mean, rgb_std=rgb_std,
                                   depth_stats=None, depth_clip=clip, train=False)
        e4 = build_e4(cfg, cat, args.rgb_checkpoint, args.depth_checkpoint,
                      args.fusion_checkpoint, device)
        cm = calib[cat]
        med_r, sc_r = cm["rgb"]["median"], cm["rgb"]["scale"]
        med_d, sc_d = cm["depth"]["median"], cm["depth"]["scale"]
        depth_norm = data_cfg.get("depth_norm", "robust_median")
        train_recs = [r for r in discover_split(root, cat, "train") if r.label == 0]
        # Deterministic TRAIN-NORMAL subset (sorted for reproducibility).
        train_recs = sorted(train_recs, key=lambda r: r.xyz_path.stem)
        srcs = [] if args.sources <= 0 else train_recs[: args.sources]

        rows = {k: [] for k in ("source", "kind", "subtype", "severity", "seed",
                                "rgb_map", "depth_map", "fused_mean", "gt_mask",
                                "valid_frac", "e_rgb_topk", "e_depth_topk",
                                "e_rgb_mean", "e_depth_mean", "p_rgb", "evidence_class",
                                "s_full", "s_rgb_only", "s_depth_only",
                                "delta_rgb", "delta_depth",
                                "spatial_pa_rgb", "spatial_pa_depth", "spatial_pa_fused")}
        for rec in tqdm(srcs, desc=f"attrib {cat}", leave=False):
            sid = f"{cat}/train/good/{rec.xyz_path.stem}"
            rgb0 = np.array(load_rgb(rec.rgb_path))
            xyz0 = load_xyz(rec.xyz_path)
            depth0, valid0 = extract_depth(xyz0), valid_mask_from_xyz(xyz0)
            for kind, subtype, sev, kw in PLAN:
                seed = det_seed(sid, kind, subtype, sev)
                rgb_p, depth_p, valid_p, mask = apply_variant(
                    rgb0, depth0, valid0, kind, subtype, seed, kw)
                rgb_t, depth_t, valid_t = tensorize(Image.fromarray(rgb_p), depth_p,
                                                    valid_p, tf, depth_stats, clip, H, W)
                v = valid_t[0, 0].numpy().astype(bool)
                gt_small = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                                    .resize((W, H), Image.NEAREST)) > 127
                with torch.no_grad():
                    out = e4.predict(rgb_t.to(device), depth_t.to(device), valid_t.to(device))
                a_r = out["rgb_map"][0, 0].cpu().numpy().astype(np.float64)
                a_d = out["depth_map"][0, 0].cpu().numpy().astype(np.float64)
                zr = calibrate(a_r, med_r, sc_r)
                zd = calibrate(a_d, med_d, sc_d)
                fused = np.where(v, 0.5 * zr + 0.5 * zd, zr)
                e_r_topk = region_evidence(zr, gt_small, mode="topk", frac=0.01)
                e_d_topk = region_evidence(zd, gt_small, mode="topk", frac=0.01)
                e_r_mean = region_evidence(zr, gt_small, mode="mean")
                e_d_mean = region_evidence(zd, gt_small, mode="mean")
                from xmvad.metrics.attribution import relative_evidence
                p_rgb, _ = relative_evidence(e_r_topk, e_d_topk)
                cls = evidence_class(e_r_topk, e_d_topk)
                s = variant_scores(fused, None)
                s_r = variant_scores(zr, None)
                s_d = variant_scores(zd, None)
                da = decision_ablation(s["max"], s_r["max"], s_d["max"])
                from xmvad.metrics.attribution import spatial_explanation_metrics
                pa_r = spatial_explanation_metrics(a_r, gt_small)["pixel_auroc"]
                pa_d = spatial_explanation_metrics(a_d, gt_small)["pixel_auroc"]
                pa_f = spatial_explanation_metrics(fused, gt_small)["pixel_auroc"]
                rows["source"].append(sid)
                rows["kind"].append(kind)
                rows["subtype"].append(subtype)
                rows["severity"].append(sev)
                rows["seed"].append(seed)
                rows["rgb_map"].append(a_r.astype(np.float32))
                rows["depth_map"].append(a_d.astype(np.float32))
                rows["fused_mean"].append(fused.astype(np.float32))
                rows["gt_mask"].append(gt_small)
                rows["valid_frac"].append(float(v.mean()) if v.size else 0.0)
                rows["e_rgb_topk"].append(e_r_topk)
                rows["e_depth_topk"].append(e_d_topk)
                rows["e_rgb_mean"].append(e_r_mean)
                rows["e_depth_mean"].append(e_d_mean)
                rows["p_rgb"].append(p_rgb)
                rows["evidence_class"].append(cls)
                rows["s_full"].append(da["S_full"])
                rows["s_rgb_only"].append(da["S_rgb_only"])
                rows["s_depth_only"].append(da["S_depth_only"])
                rows["delta_rgb"].append(da["Delta_RGB"])
                rows["delta_depth"].append(da["Delta_DEPTH"])
                rows["spatial_pa_rgb"].append(pa_r)
                rows["spatial_pa_depth"].append(pa_d)
                rows["spatial_pa_fused"].append(pa_f)

        out_dir = REPO_ROOT / "experiments" / "attribution" / "attribution_data"
        meta_dir = REPO_ROOT / "experiments" / "attribution" / "attribution_meta"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{cat}.npz",
                            source=np.array(rows["source"]),
                            kind=np.array(rows["kind"]),
                            subtype=np.array(rows["subtype"]),
                            severity=np.array(rows["severity"]),
                            seed=np.array(rows["seed"], dtype=np.int64),
                            rgb_map=np.stack(rows["rgb_map"]),
                            depth_map=np.stack(rows["depth_map"]),
                            fused_mean=np.stack(rows["fused_mean"]),
                            gt_mask=np.stack(rows["gt_mask"]),
                            valid_frac=np.array(rows["valid_frac"]),
                            e_rgb_topk=np.array(rows["e_rgb_topk"]),
                            e_depth_topk=np.array(rows["e_depth_topk"]),
                            e_rgb_mean=np.array(rows["e_rgb_mean"]),
                            e_depth_mean=np.array(rows["e_depth_mean"]),
                            p_rgb=np.array(rows["p_rgb"]),
                            evidence_class=np.array(rows["evidence_class"]),
                            s_full=np.array(rows["s_full"]),
                            s_rgb_only=np.array(rows["s_rgb_only"]),
                            s_depth_only=np.array(rows["s_depth_only"]),
                            delta_rgb=np.array(rows["delta_rgb"]),
                            delta_depth=np.array(rows["delta_depth"]),
                            spatial_pa_rgb=np.array(rows["spatial_pa_rgb"]),
                            spatial_pa_depth=np.array(rows["spatial_pa_depth"]),
                            spatial_pa_fused=np.array(rows["spatial_pa_fused"]),
                            label=np.ones(len(rows["kind"]), dtype=np.int64))
        meta = {"category": cat, "plan": [(k, st, sv, kw) for k, st, sv, kw in PLAN],
                "sources": [f"{r.category}/{r.split}/{r.defect}/{r.xyz_path.stem}"
                            for r in srcs],
                "mode": "train_normal_only_injected",
                "calibration": cm, "depth_norm_method": depth_norm,
                "seed_rule": "md5(source|kind|subtype|severity)"}
        with open(meta_dir / f"{cat}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote attribution data for {cat}: {len(rows['kind'])} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())