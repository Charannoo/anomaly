"""Distillation losses + validity-mask utilities (Phase 2).

Core discrepancy: per-pixel cosine distance between L2-normalized teacher
and student features::

    d(x, y) = 1 - cosine(T(x, y), S(x, y))   in [0, 2]

Invalid-depth handling: Phase 1 normalizes invalid depth pixels to 0.0 and
ships an explicit boolean ``valid`` mask. Losses / maps / scores must mask
invalid regions so background never dominates. Low-resolution validity masks
are derived by MAJORITY rule: a low-res pixel is valid iff at least half of
its contributing full-res pixels are valid — implemented as area
interpolation of the float mask with threshold 0.5.

Rationale note (Phase 2 hardening H2): an earlier all-valid (0.999) rule was
measured to discard absurd amounts of valid geometry (e.g. cable_gland 63%
input-valid but only ~5% surviving at 1/16; peach/potato/carrot/foam similar
because scattered invalid sensor pixels poisoned whole 16x16 cells). The
majority rule was chosen on principle — a cell's measurement is more real
than missing — NOT by tuning against test AUROC (see mask_survival.csv).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_valid_mask(
    valid: torch.Tensor, size: tuple[int, int], min_valid_frac: float = 0.5
) -> torch.Tensor:
    """Downsample a ``Bx1xHxW`` bool validity mask by majority rule.

    Returns a bool tensor of shape ``Bx1xH'xW'``. Uses area interpolation
    (fraction of valid contributors) with threshold ``min_valid_frac``
    (default 0.5: at least half the contributors must be valid).
    """
    if valid.dtype is not torch.bool:
        valid = valid.bool()
    frac = F.interpolate(valid.float(), size=size, mode="area")
    return frac >= min_valid_frac


def cosine_discrepancy(
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean ``1 - cosine`` over (valid) spatial positions. Scalar tensor.

    Args:
        teacher/student: ``BxCxHxW`` (projected) feature maps.
        valid: optional ``Bx1xHxW`` bool mask (broadcast over batch).
    """
    t = teacher / (teacher.norm(p=2, dim=1, keepdim=True) + eps)
    s = student / (student.norm(p=2, dim=1, keepdim=True) + eps)
    d = 1.0 - (t * s).sum(dim=1, keepdim=True)  # Bx1xHxW
    if valid is not None:
        v = valid.bool()
        denom = v.float().sum().clamp_min(1.0)
        return (d * v.float()).sum() / denom
    return d.mean()


def cosine_anomaly_map(
    teacher: torch.Tensor,
    student: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-pixel ``1 - cosine`` map, ``Bx1xHxW``, no reduction."""
    t = teacher / (teacher.norm(p=2, dim=1, keepdim=True) + eps)
    s = student / (student.norm(p=2, dim=1, keepdim=True) + eps)
    return 1.0 - (t * s).sum(dim=1, keepdim=True)
