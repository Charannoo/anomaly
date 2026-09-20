"""Checkpoint save/load + model-size helpers (Phase 2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def git_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, scheduler,
                    epoch: int, val_loss: float, config: dict, seed: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "val_loss": float(val_loss),
            "config": config,
            "seed": seed,
            "torch_version": torch.__version__,
            "git_hash": git_hash(),
            "modality": getattr(model, "modality", None),
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    return ckpt


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    teacher = sum(p.numel() for n, p in model.named_parameters() if "teacher" in n and "teacher_projs" not in n)
    return {"total": total, "trainable": trainable, "frozen_teacher": teacher}


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6
