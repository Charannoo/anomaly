"""Phase 2 evaluation: test-split inference + category metrics + figures.

For each category, loads ``<checkpoint>`` (a ``<category>_best.pt`` file or a
directory of them), runs the test split, and writes:

- experiments/predictions/<exp>/<category>/<id>.npz  (score, map, gt, label)
- experiments/tables/<exp>_metrics.csv               (category rows + MEAN)
- experiments/runs/<stamp>_<exp>_eval/metrics.json
- experiments/figures/overlays/<exp>/<category>/*.png (fixed 0-1 colormap)
- experiments/figures/failures/<rgb|depth>/<category>/*.png (top FP/FN)

Evaluation resolution = config input_size (default 256x256); anomaly maps and
GT are compared at that resolution (documented choice, see METRICS.md).

Examples:
    python scripts/evaluate.py --config configs/baseline_rgb.yaml --checkpoint checkpoints/rgb_baseline
    python scripts/evaluate.py --config configs/baseline_depth.yaml --checkpoint checkpoints/depth_baseline/bagel_best.pt --categories bagel
"""

from __future__ import annotations

import argparse
import datetime
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.data.depth import depth_to_uint8
from xmvad.data.mvtec3d import MVTec3DDataset, discover_categories, discover_split
from xmvad.data.transforms import SynchronizedTransform
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config
from xmvad.utils.env import collect_env

MODALITY_FIGDIR = {"rgb": "rgb", "depth": "depth"}


def resolve_checkpoints(checkpoint: str, categories: list[str]) -> dict[str, Path]:
    p = Path(checkpoint)
    out: dict[str, Path] = {}
    if p.is_file():
        if len(categories) == 1:
            return {categories[0]: p}
        raise SystemExit("A single checkpoint file requires exactly one --categories entry.")
    for cat in categories:
        cands = sorted(p.glob(f"{cat}_best.pt")) + sorted(p.glob(f"{cat}.pt"))
        if not cands:
            print(f"[warn] no checkpoint for category {cat} in {p}; skipping.")
            continue
        out[cat] = cands[0]
    return out


def build_model_from_ckpt(ckpt_path: Path, cfg: dict, device: torch.device) -> TeacherStudentAD:
    model_cfg = cfg.get("model", {})
    # Modality/params stored in checkpoint take precedence (reproducibility).
    probe = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    modality = probe.get("modality") or model_cfg.get("modality", "rgb")
    model = TeacherStudentAD(
        modality=modality,
        level_weights=tuple(model_cfg.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
        pretrained=False,  # weights come from checkpoint, not ImageNet
        score_mode=model_cfg.get("score_mode", "max"),
        topk_frac=float(model_cfg.get("topk_frac", 0.01)),
        smooth_sigma=float(model_cfg.get("smooth_sigma", 0.0)),
    )
    load_checkpoint(ckpt_path, model)
    model.to(device).eval()
    return model


def overlay_map(rgb_u8: np.ndarray, amap: np.ndarray) -> np.ndarray:
    """Jet-colored anomaly map alpha-blended onto RGB (map in [0,1]-ish)."""
    import matplotlib.cm as cm

    m = np.clip(amap, 0.0, 1.0)
    colored = (cm.jet(m)[..., :3] * 255).astype(np.uint8)
    return ((0.55 * rgb_u8.astype(float) + 0.45 * colored.astype(float))).astype(np.uint8)


def save_overlay_fig(path: Path, rgb_u8: np.ndarray, depth_u8: np.ndarray | None,
                     gt: np.ndarray | None, amap: np.ndarray, score: float,
                     title: str, modality: str) -> None:
    base = rgb_u8 if modality == "rgb" else np.stack([depth_u8] * 3, axis=-1)
    gt_show = (gt.astype(np.uint8) * 255) if gt is not None else np.zeros(base.shape[:2], dtype=np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(f"{title}  score={score:.3f}", fontsize=10)
    axes[0].imshow(base if modality == "rgb" else depth_u8, cmap=None if modality == "rgb" else "gray")
    axes[0].set_title("RGB" if modality == "rgb" else "depth")
    axes[1].imshow(gt_show, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("GT")
    axes[2].imshow(np.clip(amap, 0, 1), cmap="jet", vmin=0, vmax=1)  # FIXED scale
    axes[2].set_title("anomaly map (0-1)")
    axes[3].imshow(overlay_map(rgb_u8, amap))
    axes[3].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def infer_category(cfg: dict, category: str, ckpt_path: Path, device: torch.device,
                   exp_dir: dict[str, Path], args) -> dict:
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    modality = cfg.get("model", {}).get("modality", "rgb")

    # Depth stats: re-fit on train-normal via the shared helper
    # (deterministic, no test leakage — same stats training used).
    depth_stats = None
    if modality == "depth":
        from xmvad.data.stats import fit_train_depth_stats

        depth_stats = fit_train_depth_stats(root, category, data_cfg.get("depth_norm", "robust_median"))

    tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats, depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=False,
    )
    recs = discover_split(root, category, "test")
    ds = MVTec3DDataset(recs, input_size=input_size, transform=tf, depth_stats=depth_stats)
    loader = DataLoader(ds, batch_size=int(getattr(args, "batch_size", 0) or 8), shuffle=False,
                        num_workers=0)
    model = build_model_from_ckpt(ckpt_path, cfg, device)
    key = "rgb" if modality == "rgb" else "depth"

    ids, labels, defects, scores, maps, gts, rgbs, depths, valids = [], [], [], [], [], [], [], [], []
    for batch in tqdm(loader, desc=f"infer {category}", leave=False):
        x = batch[key].to(device)
        valid = batch["valid"].to(device) if modality == "depth" else None
        out = model.predict(x, valid)
        am = out["anomaly_map"].cpu().numpy()  # Bx1xHxW
        sc = out["score"].cpu().numpy()
        for i in range(x.shape[0]):
            idx = len(ids)
            rec = recs[idx]
            ids.append(rec.sample_id)
            labels.append(rec.label)
            defects.append(rec.defect)
            scores.append(float(sc[i]))
            maps.append(am[i, 0].astype(np.float32))
            gts.append(batch["gt"][i].numpy().astype(bool))
            valids.append(batch["valid"][i, 0].numpy().astype(bool) if modality == "depth" else None)
            rgb_u8 = ((batch["rgb"][i].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255).clip(0, 255).astype(np.uint8)
            rgbs.append(rgb_u8)
            d = batch["depth"][i, 0].numpy()
            v = batch["valid"][i, 0].numpy()
            depths.append(depth_to_uint8(d, v))

    # Save raw predictions (recomputable metrics).
    pred_dir = exp_dir["predictions"] / category
    pred_dir.mkdir(parents=True, exist_ok=True)
    variant_lists: dict[str, list[float]] = {name: [] for name, _ in SCORE_VARIANTS}
    for sid, lab, dft, sc, am, gt, va in zip(ids, labels, defects, scores, maps, gts, valids):
        safe = sid.replace("/", "_")
        vs = variant_scores(am, va)
        for name in variant_lists:
            variant_lists[name].append(vs[name])
        np.savez_compressed(pred_dir / f"{safe}.npz", score=np.float32(sc), map=am,
                            gt=gt, label=np.int64(lab), defect=dft,
                            score_max=np.float32(vs["max"]),
                            score_topk_0_1pct=np.float32(vs["topk_0.1pct"]),
                            score_topk_1pct=np.float32(vs["topk_1pct"]))

    labels_np = np.array(labels)
    metrics = {
        "category": category,
        "image_auroc_max": image_auroc(labels_np, np.array(variant_lists["max"])),
        "image_auprc_max": image_auprc(labels_np, np.array(variant_lists["max"])),
        "image_auroc_topk_0.1pct": image_auroc(labels_np, np.array(variant_lists["topk_0.1pct"])),
        "image_auprc_topk_0.1pct": image_auprc(labels_np, np.array(variant_lists["topk_0.1pct"])),
        "image_auroc_topk_1pct": image_auroc(labels_np, np.array(variant_lists["topk_1pct"])),
        "image_auprc_topk_1pct": image_auprc(labels_np, np.array(variant_lists["topk_1pct"])),
        "pixel_auroc": pixel_auroc(gts, maps),
        "pixel_auprc": pixel_auprc(gts, maps),
        "aupro": aupro(gts, maps),
        "num_test_images": len(ids),
        "num_anomalous": int(labels_np.sum()),
    }

    # Visualizations: up to K normal + K anomalous (fixed colormap 0-1).
    K = int(getattr(args, "num_viz", 10) or 10)
    normals = [i for i, l in enumerate(labels) if l == 0][:K]
    anoms = [i for i, l in enumerate(labels) if l == 1][:K]
    ov_dir = exp_dir["overlays"] / category
    ov_dir.mkdir(parents=True, exist_ok=True)
    for i in normals + anoms:
        safe = ids[i].replace("/", "_")
        save_overlay_fig(ov_dir / f"{safe}.png", rgbs[i], depths[i],
                         gts[i] if labels[i] == 1 else None, maps[i], scores[i],
                         f"{ids[i]} label={labels[i]}", modality)

    # Failure cases: top-FP (normal, highest score) + top-FN (anom, lowest).
    F = int(getattr(args, "num_failures", 5) or 5)
    fp_idx = sorted(normals_all(ids, labels, scores), key=lambda i: -scores[i])[:F]
    fn_idx = sorted(anoms_all(ids, labels, scores), key=lambda i: scores[i])[:F]
    fail_dir = exp_dir["failures"] / category
    fail_dir.mkdir(parents=True, exist_ok=True)
    for tag, idxs in (("FP", fp_idx), ("FN", fn_idx)):
        for i in idxs:
            safe = ids[i].replace("/", "_")
            save_overlay_fig(fail_dir / f"{tag}_{safe}.png", rgbs[i], depths[i], gts[i],
                             maps[i], scores[i], f"{tag} {ids[i]} label={labels[i]}", modality)

    # Development latency (NOT deployment benchmark): CPU/GPU forward timing.
    metrics_lat = dev_latency(model, key, input_size, device)
    metrics.update(metrics_lat)
    return {"metrics": metrics, "ids": ids, "labels": labels, "scores": scores,
            "maps": maps, "gts": gts}


def normals_all(ids, labels, scores):
    return [i for i, l in enumerate(labels) if l == 0]


def anoms_all(ids, labels, scores):
    return [i for i, l in enumerate(labels) if l == 1]


@torch.no_grad()
def dev_latency(model, key: str, input_size: tuple[int, int], device: torch.device,
                iters: int = 100, warmup: int = 10) -> dict:
    model.eval()
    ch = 3 if key == "rgb" else 1
    x = torch.randn(1, ch, *input_size, device=device)
    v = torch.ones(1, 1, *input_size, dtype=torch.bool, device=device) if key == "depth" else None
    for _ in range(warmup):
        model.predict(x, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(x, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"dev_latency_note": "development-only forward timing, not deployment benchmark",
            "dev_latency_mean_ms": float(a.mean()), "dev_latency_p50_ms": float(np.median(a)),
            "dev_latency_device": str(device)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Phase 2 baselines on MVTec 3D-AD test.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="category ckpt file or dir of <cat>_best.pt")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-viz", type=int, default=10)
    ap.add_argument("--num-failures", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        return 2
    cats = args.categories or cfg.get("data", {}).get("categories") or discover_categories(str(root))
    ckpts = resolve_checkpoints(args.checkpoint, cats)
    if not ckpts:
        print("No checkpoints resolved. Nothing to evaluate.")
        return 2
    dev = args.device or cfg.get("device", "auto")
    device = torch.device("cuda" if (dev == "cuda" or (dev == "auto" and torch.cuda.is_available())) else "cpu")

    exp = cfg.get("experiment", cfg.get("model", {}).get("name", "baseline"))
    modality = cfg.get("model", {}).get("modality", "rgb")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "experiments" / "runs" / f"{stamp}_{exp}_eval"
    exp_dir = {
        "predictions": REPO_ROOT / "experiments" / "predictions" / exp,
        "overlays": REPO_ROOT / "experiments" / "figures" / "overlays" / exp,
        "failures": REPO_ROOT / "experiments" / "figures" / "failures" / MODALITY_FIGDIR.get(modality, modality),
        "run": run_dir,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(collect_env({"eval_categories": sorted(ckpts), "device": str(device)}), f, indent=2)

    rows = []
    for cat, ckpt in sorted(ckpts.items()):
        print(f"[eval] {cat} <- {ckpt}")
        out = infer_category(cfg, cat, ckpt, device, exp_dir, args)
        rows.append(out["metrics"])

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in
                   ("category", "num_test_images", "num_anomalous",
                    "dev_latency_note", "dev_latency_mean_ms",
                    "dev_latency_p50_ms", "dev_latency_device")]
    mean_row = {"category": "MEAN"}
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mean_row[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    mean_row["num_test_images"] = int(df["num_test_images"].sum())
    mean_row["num_anomalous"] = int(df["num_anomalous"].sum())
    tables_dir = REPO_ROOT / "experiments" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    full.to_csv(tables_dir / f"{exp}_metrics.csv", index=False)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"experiment": exp, "eval_resolution": list(cfg.get("data", {}).get("input_size", [256, 256])),
                   "categories": rows, "mean": mean_row}, f, indent=2)
    print(full.to_string(index=False))
    print(f"[eval] wrote {tables_dir / f'{exp}_metrics.csv'} and {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
