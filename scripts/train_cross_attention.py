"""Train E5/E6/E7 XMV cross-attention (Phase 4B). Frozen Phase 2 branches.

Per category (MVTec protocol), normal samples only (train + validation
splits; NEVER the test split). Only XMV attention parameters enter the
optimizer (branches frozen). Early stopping (patience) on validation-normal
loss at E4's coefficient, best restored. Identical protocol to E4 except the
interaction module (fair comparison).

Usage:
    python scripts/train_cross_attention.py --config configs/cross_attention_rgb_from_depth.yaml
    python scripts/train_cross_attention.py --config configs/xmv_ad.yaml --categories bagel
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.data.mvtec3d import MVTec3DDataset, discover_categories, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.cross_attention import XMVModel
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config
from xmvad.utils.env import collect_env
from xmvad.utils.seed import set_seed


def load_branch(ckpt_path: Path, cfg: dict, modality: str) -> TeacherStudentAD:
    model_cfg = cfg.get("model", {})
    m = TeacherStudentAD(
        modality=modality,
        level_weights=tuple(model_cfg.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
        pretrained=False, score_mode="max", topk_frac=0.01, smooth_sigma=0.0)
    load_checkpoint(ckpt_path, m)
    return m.eval()


def _attention_entropy(a: XMVModel, mask_full: bool) -> float:
    """Mean Shannon entropy (nats) over current stored attention matrices."""
    ents = []
    for tag in ("last_attn_rgb", "last_attn_depth"):
        w = getattr(a.attn, tag)
        if w is None:
            continue
        w = w.float()
        if mask_full:
            valid_row = w.any(dim=-1)
            w = w[valid_row]
            if w.numel() == 0:
                continue
        s = -(w * torch.log(w.clamp_min(1e-12))).sum(-1).mean()
        ents.append(float(s))
    return float(np.mean(ents)) if ents else float("nan")


def train_one_category(cfg: dict, category: str, args, stamp: str,
                       rgb_ckpt: Path, depth_ckpt: Path) -> dict:
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    experiment = cfg.get("experiment", "E7_xmv_bidirectional")
    mode = str(model_cfg.get("mode", "bidirectional"))
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    seed = int(tr_cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if (args.device in ("cuda",) or
                                     (args.device in (None, "auto") and torch.cuda.is_available()))
                          and args.device != "cpu" else "cpu")

    run_dir = REPO_ROOT / "experiments" / "runs" / f"{stamp}_{experiment}_{category}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(collect_env({"category": category}), f, indent=2)

    depth_stats = fit_train_depth_stats(root, category, data_cfg.get("depth_norm", "robust_median"))

    def make_tf(train: bool):
        return SynchronizedTransform(
            input_size=input_size,
            rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
            depth_stats=depth_stats,
            depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
            train=train)

    train_recs = [r for r in discover_split(root, category, "train") if r.label == 0]
    val_recs = [r for r in discover_split(root, category, "validation") if r.label == 0]
    train_ds = MVTec3DDataset(train_recs, input_size=input_size, transform=make_tf(True),
                              depth_stats=depth_stats)
    val_ds = MVTec3DDataset(val_recs, input_size=input_size, transform=make_tf(False),
                            depth_stats=depth_stats)
    bs = int(tr_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=len(train_ds) > bs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    rgb_branch = load_branch(rgb_ckpt, {"model": {"level_weights": [0.5, 0.5]}}, "rgb")
    depth_branch = load_branch(depth_ckpt, {"model": {"level_weights": [0.5, 0.5]}}, "depth")
    xm = XMVModel(rgb_branch, depth_branch, mode=mode,
                  attn_dim=int(model_cfg.get("attention_dim", 32)),
                  num_heads=int(model_cfg.get("num_heads", 2)),
                  gamma_init=float(model_cfg.get("gamma_init", 0.0)),
                  smooth_sigma=float(model_cfg.get("smooth_sigma", 0.0))).to(device)
    n_params = xm.attn.count_parameters()
    print(f"[{category}] attention params: {n_params}", flush=True)
    opt = torch.optim.AdamW(xm.interaction_parameters(), lr=float(tr_cfg.get("lr", 1e-3)),
                            weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))
    lam = float(cfg.get("loss", {}).get("lambda_res", 1e-4))
    max_epochs = int(tr_cfg.get("epochs", 50))
    patience = int(tr_cfg.get("patience", 8)) if tr_cfg.get("early_stopping", True) else 0

    hist = {"epoch": [], "train_loss": [], "val_loss": [],
            "gamma_rd": [], "gamma_dr": [], "res_mag": [], "entropy": []}
    best_val, bad, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        xm.train(True)
        tr_sum, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{category} ep{epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            out = xm.forward_train(batch["rgb"].to(device), batch["depth"].to(device),
                                   batch["valid"].to(device), lambda_res=lam)
            out["loss"].backward()
            opt.step()
            tr_sum += float(out["loss"].item())
            n += 1
        xm.train(False)
        va_sum, vn, res_mag, ent = 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = xm.forward_train(batch["rgb"].to(device), batch["depth"].to(device),
                                       batch["valid"].to(device), lambda_res=lam)
                va_sum += float(out["loss"].item() - lam * float(out["l_res"].item()))
                res_mag += float(out["l_res"].item())
                ent += _attention_entropy(xm, mask_full=True)
                vn += 1
        tr_loss, va_loss = tr_sum / max(n, 1), va_sum / max(vn, 1)
        grd = float(xm.attn.gamma_rgb_from_depth.detach())
        gdr = float(xm.attn.gamma_depth_from_rgb.detach())
        for k, v in (("epoch", epoch), ("train_loss", tr_loss), ("val_loss", va_loss),
                     ("gamma_rd", grd), ("gamma_dr", gdr),
                     ("res_mag", res_mag / max(vn, 1)), ("entropy", ent / max(vn, 1))):
            hist[k].append(v)
        print(f"[{category}] ep{epoch}: train={tr_loss:.4f} val={va_loss:.4f} "
              f"g=({grd:.4f},{gdr:.4f}) res={res_mag / max(vn, 1):.5f} "
              f"H={ent / max(vn, 1):.3f}", flush=True)
        if va_loss < best_val:
            best_val, bad = va_loss, 0
            best_state = {k: v.cpu().clone() for k, v in xm.attn.state_dict().items()}
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"[{category}] early stopping at epoch {epoch}", flush=True)
                break
    assert best_state is not None
    xm.attn.load_state_dict(best_state)

    ckpt_dir = REPO_ROOT / "checkpoints" / experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / f"{category}_best.pt"
    torch.save({"attn_state": xm.attn.state_dict(), "mode": mode,
                "gamma_rd": float(xm.attn.gamma_rgb_from_depth.detach()),
                "gamma_dr": float(xm.attn.gamma_depth_from_rgb.detach()),
                "val_loss": best_val, "config": cfg, "seed": seed,
                "attn_params": n_params, "torch_version": torch.__version__}, dest)
    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist))
        w.writeheader()
        w.writerows([{k: hist[k][i] for k in hist} for i in range(len(hist["epoch"]))])
    # Training curves plot (loss / gammas / residual magnitude).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{category} {experiment}")
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["gamma_rd"], label="gamma_rgb<-depth")
    axes[1].plot(hist["epoch"], hist["gamma_dr"], label="gamma_depth<-rgb")
    axes[1].set_title("gammas")
    axes[1].legend()
    axes[2].plot(hist["epoch"], hist["res_mag"])
    axes[2].set_title("residual magnitude")
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=100)
    plt.close(fig)
    print(f"[{category}] best val {best_val:.4f} -> {dest}", flush=True)
    return {"category": category, "best_val": best_val,
            "gamma_rd": float(xm.attn.gamma_rgb_from_depth.detach()),
            "gamma_dr": float(xm.attn.gamma_depth_from_rgb.detach()),
            "attn_params": n_params, "checkpoint": str(dest)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xmv_ad.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        return 2
    cats = args.categories or cfg.get("data", {}).get("categories") or discover_categories(str(root))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for cat in cats:
        r = sorted(Path(args.rgb_checkpoint).glob(f"{cat}_best.pt"))
        d = sorted(Path(args.depth_checkpoint).glob(f"{cat}_best.pt"))
        if not r or not d:
            raise FileNotFoundError(f"Missing Phase 2 checkpoint for {cat}")
        train_one_category(copy.deepcopy(cfg), cat, args, stamp, r[0], d[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())