"""Train the tiny dual-head reliability gate (Phase 3C).

Data: synthetic gate features/targets from experiments/gate_data/*.npz
(train-normal derived only). Feature normalization fit on gate_train.
Class-balanced across CLEAN/RGB_ONLY/DEPTH_ONLY/JOINT. Frozen branches.
Loss: MSE(img) + lambda_map * MSE(map), lambda_map = 1.0 (pre-registered).

Writes checkpoints/E3C_gate/ (gate_best.pt, feat_norm.json, config, metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from xmvad.models.fusion.reliability_gate import CONFIDENCE_FLOOR_ALPHA, ReliabilityGate
from xmvad.utils.seed import set_seed

LAMBDA_MAP = 1.0  # pre-registered


def load_all(data_dir: Path):
    X, Wi, Wm, G, S, C = [], [], [], [], [], []
    for npz_path in sorted(data_dir.glob("*.npz")):
        cat = npz_path.stem
        z = np.load(npz_path)
        meta = json.load(open(data_dir / f"{cat}_meta.json", encoding="utf-8"))
        X.append(z["features"])
        Wi.append(z["w_img_target"])
        Wm.append(z["w_map_target"])
        G.extend(meta["groups"])
        S.extend(meta["severities"])
        C.extend([cat] * len(meta["groups"]))
    return (np.concatenate(X).astype(np.float32), np.concatenate(Wi).astype(np.float32),
            np.concatenate(Wm).astype(np.float32), np.array(G), np.array(S), np.array(C))


def dominant_metrics(w_pred: np.ndarray, w_true: np.ndarray, groups: np.ndarray):
    """Dominant-modality accuracy + macro-F1, excluding neutral CLEAN."""
    from sklearn.metrics import f1_score

    out = {}
    mask_all = groups != "clean"
    for tag, m in (("all", mask_all),
                   ("rgb_only", groups == "rgb_only"),
                   ("depth_only", groups == "depth_only"),
                   ("joint", groups == "joint")):
        if m.sum() == 0:
            continue
        yp = w_pred[m].argmax(axis=1)
        yt = w_true[m].argmax(axis=1)
        out[f"{tag}_acc"] = round(float((yp == yt).mean()), 4)
        out[f"{tag}_macroF1"] = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)
        out[f"{tag}_n"] = int(m.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/gate_data")
    ap.add_argument("--out", default="checkpoints/E3C_gate")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    X, Wi, Wm, G, S, C = load_all(REPO_ROOT / args.data)
    # gate_train/gate_val membership: recover from meta splits per category.
    is_train = []
    for cat in sorted((REPO_ROOT / args.data).glob("*_meta.json")):
        meta = json.load(open(cat, encoding="utf-8"))
        is_train.extend([s == "gate_train" for s in meta["splits"]])
    is_train = np.array(is_train)
    mu, sd = X[is_train].mean(axis=0), X[is_train].std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    # Class-balanced sampler over the 4 groups (train only).
    classes, counts = np.unique(G[is_train], return_counts=True)
    w = np.array([1.0 / dict(zip(classes, counts))[g] for g in G[is_train]])
    sampler = WeightedRandomSampler(torch.from_numpy(w / w.sum() * len(w)),
                                    num_samples=int(is_train.sum()), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xn[is_train]),
                                         torch.from_numpy(Wi[is_train]),
                                         torch.from_numpy(Wm[is_train])),
                           batch_size=args.batch_size, sampler=sampler)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xn[~is_train]),
                                         torch.from_numpy(Wi[~is_train]),
                                         torch.from_numpy(Wm[~is_train])),
                           batch_size=2048)

    model = ReliabilityGate(in_dim=X.shape[1])
    print(f"gate params: {model.count_parameters()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb_img, yb_map in tr_loader:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = mse(out["w_img"], yb_img) + LAMBDA_MAP * mse(out["w_map"], yb_map)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
        model.eval()
        va_loss, outs = 0.0, []
        with torch.no_grad():
            for xb, yb_img, yb_map in va_loader:
                out = model(xb)
                va_loss += float((mse(out["w_img"], yb_img) + LAMBDA_MAP * mse(out["w_map"], yb_map)).item())
                outs.append((out["w_img"].numpy(), out["w_map"].numpy()))
        tr_loss /= max(len(tr_loader), 1)
        va_loss /= max(len(va_loader), 1)
        print(f"epoch {epoch}: train={tr_loss:.4f} val={va_loss:.4f}", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "in_dim": X.shape[1],
                "alpha": CONFIDENCE_FLOOR_ALPHA, "lambda_map": LAMBDA_MAP,
                "seed": args.seed, "val_loss": best_val}, out_dir / "gate_best.pt")
    json.dump({"mean": mu.tolist(), "std": sd.tolist()},
              open(out_dir / "feat_norm.json", "w", encoding="utf-8"))
    json.dump({"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
               "seed": args.seed, "lambda_map": LAMBDA_MAP,
               "alpha": CONFIDENCE_FLOOR_ALPHA, "params": model.count_parameters()},
              open(out_dir / "config.json", "w", encoding="utf-8"), indent=2)

    # Synthetic validation metrics (gate_val only).
    model.eval()
    with torch.no_grad():
        full = model(torch.from_numpy(Xn[~is_train]))
    pv_img, pv_map = full["w_img"].numpy(), full["w_map"].numpy()
    gv = ~is_train
    rep = {"val_loss": round(best_val, 4),
           "img_mae": round(float(np.abs(pv_img - Wi[gv]).mean()), 4),
           "map_mae": round(float(np.abs(pv_map - Wm[gv]).mean()), 4),
           "img_corr": round(float(np.corrcoef(pv_img[:, 0], Wi[gv][:, 0])[0, 1]), 4),
           "map_corr": round(float(np.corrcoef(pv_map[:, 0], Wm[gv][:, 0])[0, 1]), 4),
           "img_dominant": dominant_metrics(pv_img, Wi[gv], G[gv]),
           "map_dominant": dominant_metrics(pv_map, Wm[gv], G[gv])}
    # By severity (non-clean).
    Sv, Gv = S[gv], G[gv]
    for sev in ("mild", "medium", "strong"):
        m = (Sv == sev)
        rep[f"img_{sev}"] = dominant_metrics(pv_img[m], Wi[gv][m], Gv[m])
        rep[f"map_{sev}"] = dominant_metrics(pv_map[m], Wm[gv][m], Gv[m])
    json.dump(rep, open(out_dir / "synth_val_metrics.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
