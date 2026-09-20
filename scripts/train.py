"""Phase 2 training entry point: E1 RGB-only / E2 depth-only baselines.

Trains ONE model per category (MVTec protocol), normal samples only.

Examples:
    python scripts/train.py --config configs/baseline_rgb.yaml
    python scripts/train.py --config configs/baseline_depth.yaml --categories bagel rope
    python scripts/train.py --config configs/baseline_rgb.yaml --categories bagel --epochs 3 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from xmvad.data.mvtec3d import MVTec3DDataset, discover_categories, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.training.checkpoints import count_parameters, file_size_mb
from xmvad.training.trainer import Trainer
from xmvad.utils.config import load_config
from xmvad.utils.env import collect_env
from xmvad.utils.seed import set_seed


def worker_seed_fn(worker_id: int, base_seed: int = 42) -> None:
    import random

    s = (base_seed + worker_id) % (2**32)
    np.random.seed(s)
    random.seed(s)


def build_optimizer(model: TeacherStudentAD, cfg: dict):
    t = cfg.get("training", {})
    name = t.get("optimizer", "adamw").lower()
    lr = float(t.get("lr", 1e-3))
    wd = float(t.get("weight_decay", 1e-4))
    params = model.trainable_parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    t = cfg.get("training", {})
    name = t.get("scheduler", "plateau").lower()
    if name in ("plateau", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    if name in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if name in ("none", "null"):
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_category(cfg: dict, category: str, args, run_stamp: str) -> dict:
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    experiment = cfg.get("experiment", model_cfg.get("name", "baseline"))
    modality = model_cfg.get("modality", "rgb")
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    seed = int(args.seed or tr_cfg.get("seed", 42))
    set_seed(seed)

    run_dir = REPO_ROOT / "experiments" / "runs" / f"{run_stamp}_{experiment}_{category}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(collect_env({"category": category, "modality": modality}), f, indent=2)

    # Depth stats: fit on TRAIN-NORMAL only (also used for RGB runs' depth
    # channel? RGB runs ignore depth; stats still saved for provenance).
    depth_stats = None
    if modality == "depth":
        depth_stats = fit_train_depth_stats(root, category, data_cfg.get("depth_norm", "robust_median"))
        with open(run_dir / "depth_stats.json", "w", encoding="utf-8") as f:
            json.dump(depth_stats, f, indent=2)

    train_recs = [r for r in discover_split(root, category, "train") if r.label == 0]
    val_recs = [r for r in discover_split(root, category, "validation") if r.label == 0]
    if not train_recs:
        raise RuntimeError(f"No train-normal records for {category} under {root}.")
    train_tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats,
        depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=True,
    )
    val_tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats,
        depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=False,
    )
    # NOTE: MVTec3DDataset binds per-category stats via with_depth_stats when
    # depth_stats is a {category: stats} dict; here we pass the single stats
    # dict directly (or None for RGB).
    train_ds = MVTec3DDataset(train_recs, input_size=input_size, transform=train_tf,
                              depth_stats=depth_stats)
    val_ds = MVTec3DDataset(val_recs, input_size=input_size, transform=val_tf,
                            depth_stats=depth_stats) if val_recs else None
    bs = int(args.batch_size or tr_cfg.get("batch_size", 8))
    nw = int(data_cfg.get("num_workers", 0 if sys.platform == "win32" else 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=False, drop_last=len(train_ds) > bs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw) if val_ds else None

    lw = tuple(model_cfg.get("level_weights", [0.5, 0.5]))
    model = TeacherStudentAD(
        modality=modality,
        level_weights=lw,  # type: ignore[arg-type]
        pretrained=bool(model_cfg.get("pretrained", True)),
        score_mode=model_cfg.get("score_mode", "max"),
        topk_frac=float(model_cfg.get("topk_frac", 0.01)),
        smooth_sigma=float(model_cfg.get("smooth_sigma", 0.0)),
    )
    info = count_parameters(model)
    print(f"[{category}/{modality}] params: {info} | train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    trainer = Trainer(
        model, optimizer, scheduler, train_loader, val_loader,
        device=args.device or cfg.get("device", "auto"),
        run_dir=run_dir, config=cfg, seed=seed,
        max_epochs=int(args.epochs or tr_cfg.get("epochs", 50)),
        early_stopping_patience=int(tr_cfg.get("early_stopping_patience", 0)),
        use_amp=bool(tr_cfg.get("amp", False)),
        use_tensorboard=bool(tr_cfg.get("tensorboard", False)),
    )
    if args.resume:
        trainer.resume(args.resume)
    best_ckpt = trainer.train()

    # Publish checkpoint + model info.
    ckpt_dir = REPO_ROOT / "checkpoints" / experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / f"{category}_best.pt"
    import shutil

    shutil.copyfile(best_ckpt, dest)
    with open(run_dir / "checkpoint_reference.txt", "w", encoding="utf-8") as f:
        f.write(str(dest.relative_to(REPO_ROOT)) + "\n")
    model_info = {**info, "checkpoint_mb": file_size_mb(dest),
                  "checkpoint": str(dest.relative_to(REPO_ROOT))}
    with open(run_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    print(f"[{category}/{modality}] best -> {dest} ({model_info['checkpoint_mb']:.1f} MB)")
    return {"category": category, "run_dir": str(run_dir), "checkpoint": str(dest), **model_info}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Phase 2 single-modal baselines (normal-only).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        print("Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD")
        print("dataset page and extract/copy it to data/raw/mvtec3d/.")
        return 2
    cats = args.categories or cfg.get("data", {}).get("categories") or discover_categories(str(root))
    print(f"Training {cfg.get('experiment')} on categories: {cats}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for cat in cats:
        results.append(train_one_category(copy.deepcopy(cfg), cat, args, stamp))
    print("Done. Checkpoints:")
    for r in results:
        print(f"  {r['category']}: {r['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
