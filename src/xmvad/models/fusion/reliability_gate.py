"""Tiny dual-head reliability gate (Phase 3C).

Maps a compact feature vector of frozen-branch statistics to two independent
modality-weight pairs: one for image-level detection, one for localization.
Deliberately tiny MLP — no transformers, no convolutions.

    features (D)
      -> Linear(D, 32) -> ReLU -> Dropout(p, train only)
      -> Linear(32, 16) -> ReLU
      -> detection head Linear(16, 2) -> Softmax  -> (w_rgb_img, w_depth_img)
      -> localization head Linear(16, 2) -> Softmax -> (w_rgb_map, w_depth_map)

A confidence floor (alpha, pre-registered 0.1) mixes raw softmax weights
toward uniform inside forward(), so training and inference see identical
weights: w_final = (1-alpha)*w + alpha*0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CONFIDENCE_FLOOR_ALPHA = 0.1  # pre-registered; do not sweep on real test AUROC


class ReliabilityGate(nn.Module):
    """Dual-head MLP gate. All parameters trainable (frozen branches stay out)."""

    def __init__(self, in_dim: int, hidden: tuple[int, int] = (32, 16),
                 dropout: float = 0.1, alpha: float = CONFIDENCE_FLOOR_ALPHA) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.det_head = nn.Linear(hidden[1], 2)
        self.loc_head = nn.Linear(hidden[1], 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        w_img = torch.softmax(self.det_head(h), dim=-1)
        w_map = torch.softmax(self.loc_head(h), dim=-1)
        a = self.alpha
        w_img = (1 - a) * w_img + a * 0.5
        w_map = (1 - a) * w_map + a * 0.5
        return {"w_img": w_img, "w_map": w_map}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
