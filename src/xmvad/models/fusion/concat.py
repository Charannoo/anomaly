"""E4 lightweight residual feature concatenation (Phase 4A).

Fusion at 1/16 resolution ONLY. Frozen Phase 2 student features
S_rgb/S_depth (48 ch each) are projected to d=32, concatenated, passed
through a tiny block, and decoded into per-modality residuals added back
with trainable scalar gates (init 0 => Phase 2 identity at start):

    S_rgb_fused   = S_rgb   + gamma_rgb   * delta_rgb
    S_depth_fused = S_depth + gamma_depth * delta_depth

Modality preservation: E4 still outputs independent A_RGB / A_DEPTH maps;
only the final score fusion (Phase 3A calibrated 50/50, applied later in
evaluation) combines them. The ONLY trainable parameters in E4 belong to
ConcatFusion (teachers + students frozen).
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
from xmvad.models.anomaly.teacher_student import gaussian_blur
from xmvad.models.encoders.rgb_encoder import LEVELS

IN_CH = 48  # Phase 2 stage16 channels (both modalities)


class ConcatFusion(nn.Module):
    """Tiny concat fusion block with zero-init residual gates."""

    def __init__(self, proj_dim: int = 32, gamma_init: float = 0.0) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.proj_rgb = nn.Conv2d(IN_CH, proj_dim, 1)
        self.proj_depth = nn.Conv2d(IN_CH, proj_dim, 1)
        self.mix = nn.Sequential(
            nn.Conv2d(2 * proj_dim, proj_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 3, padding=1, groups=proj_dim, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.head_rgb = nn.Conv2d(proj_dim, IN_CH, 1)
        self.head_depth = nn.Conv2d(proj_dim, IN_CH, 1)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_depth = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, s_rgb: torch.Tensor, s_depth: torch.Tensor):
        p = torch.cat([self.proj_rgb(s_rgb), self.proj_depth(s_depth)], dim=1)
        h = self.mix(p)
        d_rgb = self.head_rgb(h)
        d_depth = self.head_depth(h)
        return (s_rgb + self.gamma_rgb * d_rgb,
                s_depth + self.gamma_depth * d_depth,
                d_rgb, d_depth)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class E4ConcatModel(nn.Module):
    """Phase 2 RGB+Depth branches (frozen) + trainable 1/16 concat fusion.

    Args:
        rgb_model/depth_model: Phase 2 TeacherStudentAD instances (already
            weight-loaded; this wrapper freezes ALL their parameters).
    """

    def __init__(self, rgb_model: nn.Module, depth_model: nn.Module,
                 proj_dim: int = 32, gamma_init: float = 0.0,
                 smooth_sigma: float = 0.0) -> None:
        super().__init__()
        self.rgb = rgb_model
        self.depth = depth_model
        for p in self.rgb.parameters():
            p.requires_grad_(False)
        for p in self.depth.parameters():
            p.requires_grad_(False)
        self.rgb.train(False)
        self.depth.train(False)
        self.fusion = ConcatFusion(proj_dim=proj_dim, gamma_init=gamma_init)
        self.smooth_sigma = smooth_sigma

    def train(self, mode: bool = True) -> "E4ConcatModel":  # type: ignore[override]
        # Branches stay frozen-eval; fusion follows the requested mode.
        super().train(mode)
        self.rgb.train(False)
        self.depth.train(False)
        self.fusion.train(mode)
        return self

    def fusion_parameters(self):
        return list(self.fusion.parameters())

    def _student_taps(self, student: nn.Module, x: torch.Tensor):
        """Single manual pass returning (head8_out, prehead16). Parameters are
        frozen; activations keep grad_fn so fusion receives gradients."""
        f = student.stem(x)
        f = student.s1(f)
        f8 = student.s2(f)
        f16 = student.s3(f8)
        return student.head8(f8), student.head16(f16)

    def forward_16(self, x_rgb: torch.Tensor, x_depth: torch.Tensor,
                   valid: torch.Tensor | None = None) -> dict:
        """Stage16 fused student features + raw student stage8 features."""
        with torch.no_grad():
            t_rgb = self.rgb.teacher(x_rgb)
            t_depth = self.depth.teacher(x_depth)
        s_rgb8, h_rgb16 = self._student_taps(self.rgb.student, x_rgb)
        s_depth8, h_depth16 = self._student_taps(self.depth.student, x_depth)
        f_rgb16, f_depth16, d_rgb, d_depth = self.fusion(h_rgb16, h_depth16)
        return {"t_rgb": t_rgb, "t_depth": t_depth,
                "t_rgb16": t_rgb["stage16"], "t_depth16": t_depth["stage16"],
                "s_rgb8": s_rgb8, "s_depth8": s_depth8,
                "f_rgb16": f_rgb16, "f_depth16": f_depth16,
                "d_rgb": d_rgb, "d_depth": d_depth}

    def forward_train(self, x_rgb: torch.Tensor, x_depth: torch.Tensor,
                      valid: torch.Tensor | None = None,
                      lambda_res: float = 1e-4) -> dict[str, torch.Tensor]:
        out = self.forward_16(x_rgb, x_depth, valid)
        v16 = downsample_valid_mask(valid, out["f_rgb16"].shape[-2:]) if valid is not None else None
        l_rgb = cosine_discrepancy(out["t_rgb16"], out["f_rgb16"], None)
        l_depth = cosine_discrepancy(out["t_depth16"], out["f_depth16"], v16)
        l_res = out["d_rgb"].abs().mean() + out["d_depth"].abs().mean()
        total = l_rgb + l_depth + lambda_res * l_res
        return {"loss": total, "l_rgb16": l_rgb, "l_depth16": l_depth, "l_res": l_res}

    @torch.no_grad()
    def predict(self, x_rgb: torch.Tensor, x_depth: torch.Tensor,
                valid: torch.Tensor | None = None,
                out_size: tuple[int, int] | None = None) -> dict:
        """A_RGB_E4 / A_DEPTH_E4 at input resolution (Phase 2 map protocol)."""
        h, w = x_rgb.shape[-2:]
        out_size = out_size or (h, w)
        fw = self.forward_16(x_rgb, x_depth, valid)
        # Stage8 pathway identical to Phase 2 (raw students).
        t_rgb8 = fw["t_rgb"]["stage8"]
        t_depth8 = fw["t_depth"]["stage8"]
        maps = {}
        for tag, t8, s8, t16, f16 in (
                ("rgb", t_rgb8, fw["s_rgb8"], fw["t_rgb16"], fw["f_rgb16"]),
                ("depth", t_depth8, fw["s_depth8"], fw["t_depth16"], fw["f_depth16"])):
            m8 = cosine_anomaly_map(t8, s8).clamp_min(0.0)
            m16 = cosine_anomaly_map(t16, f16).clamp_min(0.0)
            if tag == "depth" and valid is not None:
                v8 = downsample_valid_mask(valid, m8.shape[-2:])
                v16 = downsample_valid_mask(valid, m16.shape[-2:])
                m8 = torch.where(v8, m8, torch.zeros_like(m8))
                m16 = torch.where(v16, m16, torch.zeros_like(m16))
            m8 = F.interpolate(m8, size=out_size, mode="bilinear", align_corners=False)
            m16 = F.interpolate(m16, size=out_size, mode="bilinear", align_corners=False)
            agg = 0.5 * m8 + 0.5 * m16
            agg = gaussian_blur(agg, self.smooth_sigma)
            if tag == "depth" and valid is not None:
                v = F.interpolate(valid.float(), size=out_size, mode="nearest").bool()
                agg = torch.where(v, agg, torch.zeros_like(agg))
            maps[tag] = agg
        return {"rgb_map": maps["rgb"], "depth_map": maps["depth"]}
