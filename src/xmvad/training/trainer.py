"""Generic single-modal trainer (Phase 2).

Features:
- device: 'auto' | 'cpu' | 'cuda' (auto picks cuda iff available)
- AMP (torch.amp) ONLY on CUDA; silently disabled on CPU with a log line
- deterministic seeding (caller seeds; DataLoader workers seeded)
- best (val loss) + latest checkpoints, resume, early stopping
- optional TensorBoard (SummaryWriter) if installed + enabled
- per-epoch CSV logging (train_loss, val_loss, lr)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.training.checkpoints import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: str = "auto",
        run_dir: str | Path = "experiments/runs/debug",
        config: dict | None = None,
        seed: int = 42,
        max_epochs: int = 50,
        early_stopping_patience: int = 0,  # 0 = disabled
        use_amp: bool = False,
        use_tensorboard: bool = False,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self.config = config or {}
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.log_interval = log_interval

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("[trainer] CUDA requested but unavailable — falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        if use_amp and self.device.type != "cuda":
            print("[trainer] AMP requested on CPU — disabled (CUDA-only).")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.model.to(self.device)

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.run_dir / "logs" / "tb"))
            except ImportError:
                print("[trainer] tensorboard not installed — continuing without it.")

        self.csv_path = self.run_dir / "training.csv"
        self._csv_init = False
        self.start_epoch = 0
        self.best_val = float("inf")
        self._bad_epochs = 0

    # -- checkpointing -------------------------------------------------
    def _ckpt_paths(self):
        return self.run_dir / "best.pt", self.run_dir / "latest.pt"

    def save(self, epoch: int, val_loss: float, best: bool) -> None:
        best_p, latest_p = self._ckpt_paths()
        save_checkpoint(latest_p, self.model, self.optimizer, self.scheduler, epoch, val_loss, self.config, self.seed)
        if best:
            save_checkpoint(best_p, self.model, self.optimizer, self.scheduler, epoch, val_loss, self.config, self.seed)

    def resume(self, path: str | Path) -> None:
        from xmvad.training.checkpoints import load_checkpoint

        ckpt = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val = float(ckpt.get("val_loss", float("inf")))
        print(f"[trainer] resumed {path} at epoch {self.start_epoch} (val {self.best_val:.4f})")

    # -- loops ---------------------------------------------------------
    def _modality_key(self) -> str:
        return getattr(self.model, "modality", "rgb")

    def _batch_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        key = self._modality_key()
        x = batch["rgb"] if key == "rgb" else batch["depth"]
        valid = batch.get("valid") if key == "depth" else None
        x = x.to(self.device, non_blocking=True)
        if valid is not None:
            valid = valid.to(self.device, non_blocking=True)
        return x, valid

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.model.teacher.train(False)  # teacher stays frozen-eval
        running, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            x, valid = self._batch_inputs(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.autocast("cuda"):
                    out = self.model.forward_train(x, valid)
                assert self.scaler is not None
                self.scaler.scale(out["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model.forward_train(x, valid)
                out["loss"].backward()
                self.optimizer.step()
            loss = float(out["loss"].item())
            running += loss
            n += 1
            if i % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")
        return running / max(n, 1)

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        running, n = 0.0, 0
        for batch in self.val_loader:
            x, valid = self._batch_inputs(batch)
            out = self.model.forward_train(x, valid)
            running += float(out["loss"].item())
            n += 1
        return running / max(n, 1)

    def _log_csv(self, epoch: int, train_loss: float, val_loss: float, lr: float, secs: float) -> None:
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "secs"])
            if new:
                w.writeheader()
            w.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "secs": secs})

    def train(self) -> Path:
        best_p, _ = self._ckpt_paths()
        for epoch in range(self.start_epoch, self.max_epochs):
            t0 = time.time()
            tr = self.train_epoch(epoch)
            va = self.validate()
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(va if va == va else tr)  # NaN-safe
                else:
                    self.scheduler.step()
            lr = float(self.optimizer.param_groups[0]["lr"])
            self._log_csv(epoch, tr, va, lr, time.time() - t0)
            if self.writer is not None:
                self.writer.add_scalar("loss/train", tr, epoch)
                self.writer.add_scalar("loss/val", va, epoch)
                self.writer.add_scalar("lr", lr, epoch)
            improved = va < self.best_val if va == va else False
            if improved:
                self.best_val = va
                self._bad_epochs = 0
            else:
                self._bad_epochs += 1
            self.save(epoch, va, best=improved)
            print(f"[trainer] epoch {epoch}: train={tr:.4f} val={va:.4f} lr={lr:.2e} best={self.best_val:.4f}")
            if self.patience > 0 and self._bad_epochs >= self.patience:
                print(f"[trainer] early stopping after {self.patience} epochs without improvement.")
                break
        if self.writer is not None:
            self.writer.close()
        return best_p
