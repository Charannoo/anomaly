"""Single-modal teacher/student anomaly detectors (Phase 2 baselines).

E1 RGB-only and E2 depth-only share this implementation with INDEPENDENT
weights — no multimodal interaction exists in Phase 2.

Architecture per modality (HARDENED: fixed teacher targets)::

    input (RGB 3ch | depth 1ch)
      |-- frozen teacher (MobileNetV3-Small taps stage8/stage16)
      |     -> T_8 (24 ch, frozen), T_16 (48 ch, frozen)   [NO projection]
      |-- compact student CNN (trainable)
            -> 1x1 heads -> S_8 (24 ch), S_16 (48 ch)

The teacher side is a FIXED target: no trainable projection sits between the
frozen backbone and the loss, so the target cannot drift toward whatever
makes student matching easier. Cosine discrepancy is computed directly
against the frozen teacher representations. This is lighter (fewer params)
and easier to defend.

Training (normal samples only)::

    L = w8 * mean(1 - cos(T_8, S_8)) + w16 * mean(1 - cos(T_16, S_16))

Inference::

    A_l = upsample(1 - cos(T_l, S_l));  A = w8*A_8 + w16*A_16
    optional Gaussian smoothing (sigma from config, 0 = off)
    depth: invalid cells zeroed BEFORE upsampling (same conservative mask
    as training), invalid pixels forced to 0 AFTER aggregation
    score = max(A) | top-k mean over valid region
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmvad.losses.distillation import (
    cosine_anomaly_map,
    cosine_discrepancy,
    downsample_valid_mask,
)
from xmvad.models.encoders.depth_encoder import DepthTeacher
from xmvad.models.encoders.rgb_encoder import LEVELS, OUT_CHANNELS, RGBTeacher

MODALITIES = ("rgb", "depth")


class DSConv(nn.Module):
    """Depthwise-separable 3x3 conv + BN + ReLU (lightweight student block)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class CompactStudent(nn.Module):
    """Compact fully-trainable student predicting teacher taps from raw input.

    Strides mirror the teacher: taps at 1/8 (64 ch base) and 1/16 (96 ch
    base), each projected by a trainable 1x1 head to the FROZEN teacher
    channel count (24 @ stage8, 48 @ stage16). Works for any input size
    divisible by 16.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = DSConv(in_channels, 32, stride=2)  # 1/2
        self.s1 = DSConv(32, 48, stride=2)  # 1/4
        self.s2 = DSConv(48, 64, stride=2)  # 1/8  <- tap
        self.s3 = DSConv(64, 96, stride=2)  # 1/16 <- tap
        self.head8 = nn.Conv2d(64, OUT_CHANNELS["stage8"], 1)
        self.head16 = nn.Conv2d(96, OUT_CHANNELS["stage16"], 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.s1(x)
        f8 = self.s2(x)
        f16 = self.s3(f8)
        return {"stage8": self.head8(f8), "stage16": self.head16(f16)}


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on ``Bx1xHxW``; sigma<=0 returns input."""
    if sigma <= 0:
        return x
    r = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()
    kh = k1d.view(1, 1, 1, -1)  # horizontal
    kv = k1d.view(1, 1, -1, 1)  # vertical
    x = F.conv2d(x, kh, padding=(0, r))
    x = F.conv2d(x, kv, padding=(r, 0))
    return x


class TeacherStudentAD(nn.Module):
    """One single-modal baseline model (modality = 'rgb' | 'depth')."""

    def __init__(
        self,
        modality: str = "rgb",
        level_weights: tuple[float, float] = (0.5, 0.5),
        pretrained: bool = True,
        score_mode: str = "max",  # 'max' | 'topk'
        topk_frac: float = 0.01,
        smooth_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        if score_mode not in ("max", "topk"):
            raise ValueError("score_mode must be 'max' or 'topk'")
        self.modality = modality
        self.w8, self.w16 = level_weights
        self.score_mode = score_mode
        self.topk_frac = topk_frac
        self.smooth_sigma = smooth_sigma
        self.in_channels = 3 if modality == "rgb" else 1

        self.teacher = RGBTeacher(pretrained=pretrained) if modality == "rgb" else DepthTeacher(pretrained=pretrained)
        self.student = CompactStudent(self.in_channels)

    # -- training ------------------------------------------------------
    def forward_train(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            t = self.teacher(x)  # frozen targets, no projection
        s = self.student(x)
        losses = {}
        for lvl in LEVELS:
            v = downsample_valid_mask(valid, t[lvl].shape[-2:]) if valid is not None else None
            losses[lvl] = cosine_discrepancy(t[lvl], s[lvl], v)
        total = self.w8 * losses["stage8"] + self.w16 * losses["stage16"]
        return {"loss": total, "l_stage8": losses["stage8"], "l_stage16": losses["stage16"]}

    # -- inference -----------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, valid: torch.Tensor | None = None, out_size: tuple[int, int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return anomaly maps at input resolution + image scores.

        Returns dict with ``anomaly_map`` (Bx1xHxW, >=0), ``score`` (B,),
        ``maps`` per-level upsampled maps, all finite.
        """
        h, w = x.shape[-2:]
        out_size = out_size or (h, w)
        t = self.teacher(x)  # frozen targets, no projection
        s = self.student(x)
        maps = {}
        for lvl in LEVELS:
            m = cosine_anomaly_map(t[lvl], s[lvl]).clamp_min(0.0)
            if valid is not None:
                # Mask invalid cells BEFORE upsampling with the SAME
                # conservative mask used in training. Cells excluded from the
                # loss carry untrained (arbitrary) values; bilinear upsampling
                # would otherwise bleed them into neighboring valid pixels
                # and dominate the max/top-k image score on every image.
                v_low = downsample_valid_mask(valid, m.shape[-2:])
                m = torch.where(v_low, m, torch.zeros_like(m))
            maps[lvl] = F.interpolate(m, size=out_size, mode="bilinear", align_corners=False)
        agg = self.w8 * maps["stage8"] + self.w16 * maps["stage16"]
        agg = gaussian_blur(agg, self.smooth_sigma)
        if valid is not None:
            v = F.interpolate(valid.float(), size=out_size, mode="nearest").bool()
            agg = torch.where(v, agg, torch.zeros_like(agg))
        else:
            v = None
        return {"anomaly_map": agg, "score": self._image_score(agg, v), "maps": maps}

    def _image_score(self, agg: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        flat = agg.flatten(1)  # BxN
        if valid is not None:
            v = valid.flatten(1).bool()
            # Mask invalid with -inf so max/topk ignore them; rows fully
            # invalid (should not happen) fall back to 0.
            neg = torch.full_like(flat, float("-inf"))
            masked = torch.where(v, flat, neg)
            if self.score_mode == "max":
                sc = masked.amax(dim=1)
                return torch.where(torch.isfinite(sc), sc, torch.zeros_like(sc))
            k = max(1, int(round(self.topk_frac * flat.shape[1])))
            topk = masked.topk(k, dim=1).values
            topk = torch.where(torch.isfinite(topk), topk, torch.zeros_like(topk))
            return topk.mean(dim=1)
        if self.score_mode == "max":
            return flat.amax(dim=1)
        k = max(1, int(round(self.topk_frac * flat.shape[1])))
        return flat.topk(k, dim=1).values.mean(dim=1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
