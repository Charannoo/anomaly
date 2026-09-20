"""XMV directional cross-attention (Phase 4B: E5/E6/E7).

One generic module; configuration selects the active direction(s):
- E5 ``rgb_from_depth``: RGB queries Depth only.
- E6 ``depth_from_rgb``: Depth queries RGB only.
- E7 ``bidirectional``: both (the proposed XMV interaction block).

Design (pre-registered): attention_dim=32, heads=2 (head_dim=16), fusion at
1/16 only (N=256 tokens at 256px input — direct attention, no windowing).
Per-modality 1x1 projection (48->32) + LayerNorm INSIDE the attention path
only, so gamma=0 reproduces Phase 2 bit-exactly. NO self-attention, NO FFN,
NO transformer stacks, NO positional parameters — RGB/depth are spatially
registered, so aligned token ordering is the position signal (documented).

Residuals mirror E4: S_x = S + gamma * delta, gamma init 0.

Validity (mandatory): RGB<-Depth masks invalid depth KEYS pre-softmax
(additive -inf, numerically guarded; samples with zero valid keys get zero
residual). Depth<-RGB masks delta_depth at invalid QUERY locations
post-attention. Majority-downsampled stage16 mask (approved rule).

The module stores the last attention matrices (detached) as
``last_attn_rgb`` / ``last_attn_depth`` (B,H,N,N) for interaction
visualization — labeled CROSS-MODAL INTERACTION, never explanation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmvad.losses.distillation import (
    cosine_anomaly_map,
    cosine_discrepancy,
    downsample_valid_mask,
)
from xmvad.models.anomaly.teacher_student import gaussian_blur

IN_CH = 48  # Phase 2 stage16 channels
ATTN_DIM = 32
NUM_HEADS = 2
HEAD_DIM = ATTN_DIM // NUM_HEADS
NEG_INF = -1e9

MODES = ("rgb_from_depth", "depth_from_rgb", "bidirectional")


class XMVAttention(nn.Module):
    """Single directional/bidirectional cross-attention interaction block."""

    def __init__(self, attn_dim: int = ATTN_DIM, num_heads: int = NUM_HEADS,
                 mode: str = "bidirectional", gamma_init: float = 0.0) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.mode = mode
        self.proj_rgb = nn.Conv2d(IN_CH, attn_dim, 1)
        self.proj_depth = nn.Conv2d(IN_CH, attn_dim, 1)
        self.ln_rgb = nn.LayerNorm(attn_dim)
        self.ln_depth = nn.LayerNorm(attn_dim)
        # Direction RGB <- Depth.
        self.wq_rgb = nn.Linear(attn_dim, attn_dim)
        self.wk_depth = nn.Linear(attn_dim, attn_dim)
        self.wv_depth = nn.Linear(attn_dim, attn_dim)
        self.wo_rgb = nn.Linear(attn_dim, IN_CH)
        # Direction Depth <- RGB.
        self.wq_depth = nn.Linear(attn_dim, attn_dim)
        self.wk_rgb = nn.Linear(attn_dim, attn_dim)
        self.wv_rgb = nn.Linear(attn_dim, attn_dim)
        self.wo_depth = nn.Linear(attn_dim, IN_CH)
        self.gamma_rgb_from_depth = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_depth_from_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_attn_rgb = None  # (B,H,N,N) detached, RGB<-Depth
        self.last_attn_depth = None  # (B,H,N,N) detached, Depth<-RGB

    # -- helpers ------------------------------------------------------
    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * self.head_dim)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention with optional boolean key mask.

        key_valid: (B,1,1,Nk) bool. Fully-masked rows are guarded downstream.
        Returns (context (B,H,Nq,Dh), weights (B,H,Nq,Nk)).
        """
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        row_valid = None
        if key_valid is not None:
            logits = torch.where(key_valid, logits,
                                 torch.full_like(logits, NEG_INF))
            row_valid = key_valid.any(dim=-1)  # (B,1,1): rows with >=1 valid key
        weights = torch.softmax(logits, dim=-1)
        # Guard: rows with no valid key get EXACTLY zero attention (the caller
        # separately zeroes their residual so no bias leaks in). Never NaN.
        if row_valid is not None:
            weights = weights * row_valid.unsqueeze(-1).float()
        weights = torch.where(torch.isfinite(weights), weights,
                              torch.zeros_like(weights))
        return torch.matmul(weights, v), weights

    def _direction(self, p_q: torch.Tensor, p_kv: torch.Tensor,
                   wq: nn.Linear, wk: nn.Linear, wv: nn.Linear, wo: nn.Linear,
                   key_valid: torch.Tensor | None, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = p_q.shape
        H16, W16 = hw
        q = self._heads(wq(p_q))
        k = self._heads(wk(p_kv))
        v = self._heads(wv(p_kv))
        ctx, w = self._attend(q, k, v, key_valid)
        return wo(self._merge(ctx)).view(B, H16, W16, -1).permute(0, 3, 1, 2), w

    # -- forward ------------------------------------------------------
    def forward(self, s_rgb: torch.Tensor, s_depth: torch.Tensor,
                valid16: torch.Tensor | None = None):
        """Returns (Sf_rgb, Sf_depth, d_rgb|None, d_depth|None).

        valid16: (B,1,H16,W16) bool majority mask or None.
        """
        B, _, H16, W16 = s_rgb.shape
        N = H16 * W16
        p_rgb = self.ln_rgb(self.proj_rgb(s_rgb).flatten(2).transpose(1, 2))
        p_depth = self.ln_depth(self.proj_depth(s_depth).flatten(2).transpose(1, 2))
        key_mask = None
        if valid16 is not None:
            key_mask = valid16.reshape(B, 1, 1, N)
        d_rgb = d_depth = None
        f_rgb, f_depth = s_rgb, s_depth
        self.last_attn_rgb = None
        self.last_attn_depth = None
        if self.mode in ("rgb_from_depth", "bidirectional"):
            d, w = self._direction(p_rgb, p_depth, self.wq_rgb, self.wk_depth,
                                   self.wv_depth, self.wo_rgb, key_mask, (H16, W16))
            self.last_attn_rgb = w.detach()
            if valid16 is not None:
                # Samples with zero valid depth keys: zero residual (no NaNs).
                no_valid = ~valid16.reshape(B, -1).any(dim=-1).view(B, 1, 1, 1)
                d = torch.where(no_valid, torch.zeros_like(d), d)
            d_rgb = d
            f_rgb = s_rgb + self.gamma_rgb_from_depth * d_rgb
        if self.mode in ("depth_from_rgb", "bidirectional"):
            d, w = self._direction(p_depth, p_rgb, self.wq_depth, self.wk_rgb,
                                   self.wv_rgb, self.wo_depth, None, (H16, W16))
            self.last_attn_depth = w.detach()
            if valid16 is not None:
                # Invalid query locations receive no update.
                qm = valid16.expand(-1, IN_CH, -1, -1)
                d = torch.where(qm, d, torch.zeros_like(d))
            d_depth = d
            f_depth = s_depth + self.gamma_depth_from_rgb * d_depth
        return f_rgb, f_depth, d_rgb, d_depth

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameters(self):
        """Parameters of the ACTIVE direction(s) + shared projections/norms.

        Used by the direction-disabled grad test: in E5, depth_from_rgb
        parameters must receive no gradients (and vice versa for E6).
        """
        per_dir = {"rgb_from_depth": ("wq_rgb", "wk_depth", "wv_depth", "wo_rgb",
                                      "gamma_rgb_from_depth"),
                   "depth_from_rgb": ("wq_depth", "wk_rgb", "wv_rgb", "wo_depth",
                                      "gamma_depth_from_rgb")}
        if self.mode == "bidirectional":
            keep = set(per_dir["rgb_from_depth"]) | set(per_dir["depth_from_rgb"])
        else:
            keep = set(per_dir[self.mode])
        keep |= {"proj_rgb", "proj_depth", "ln_rgb", "ln_depth"}
        return [p for n, p in self.named_parameters() if n.split(".")[0] in keep]

    def inactive_parameters(self):
        """Parameters of DISABLED directions (E5/E6); empty for E7."""
        if self.mode == "bidirectional":
            return []
        other = "depth_from_rgb" if self.mode == "rgb_from_depth" else "rgb_from_depth"
        per_dir = {"rgb_from_depth": ("wq_rgb", "wk_depth", "wv_depth", "wo_rgb",
                                      "gamma_rgb_from_depth"),
                   "depth_from_rgb": ("wq_depth", "wk_rgb", "wv_rgb", "wo_depth",
                                      "gamma_depth_from_rgb")}
        keep = set(per_dir[other])
        return [p for n, p in self.named_parameters() if n.split(".")[0] in keep]


class XMVModel(nn.Module):
    """Phase 2 RGB+Depth branches (frozen) + XMV interaction at 1/16.

    Same training/eval protocol as E4ConcatModel; only the interaction
    module differs (fair comparison).
    """

    def __init__(self, rgb_model: nn.Module, depth_model: nn.Module,
                 mode: str = "bidirectional", attn_dim: int = ATTN_DIM,
                 num_heads: int = NUM_HEADS, gamma_init: float = 0.0,
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
        self.attn = XMVAttention(attn_dim=attn_dim, num_heads=num_heads,
                                 mode=mode, gamma_init=gamma_init)
        self.smooth_sigma = smooth_sigma
        self.last_deltas = {"rgb": None, "depth": None}

    def train(self, mode: bool = True) -> "XMVModel":  # type: ignore[override]
        super().train(mode)
        self.rgb.train(False)
        self.depth.train(False)
        self.attn.train(mode)
        return self

    def interaction_parameters(self):
        return list(self.attn.parameters())

    def _student_taps(self, student: nn.Module, x: torch.Tensor):
        f = student.stem(x)
        f = student.s1(f)
        f8 = student.s2(f)
        f16 = student.s3(f8)
        return student.head8(f8), student.head16(f16)

    def forward_16(self, x_rgb: torch.Tensor, x_depth: torch.Tensor,
                   valid: torch.Tensor | None = None) -> dict:
        with torch.no_grad():
            t_rgb = self.rgb.teacher(x_rgb)
            t_depth = self.depth.teacher(x_depth)
        s_rgb8, h_rgb16 = self._student_taps(self.rgb.student, x_rgb)
        s_depth8, h_depth16 = self._student_taps(self.depth.student, x_depth)
        v16 = downsample_valid_mask(valid, h_rgb16.shape[-2:]) if valid is not None else None
        f_rgb16, f_depth16, d_rgb, d_depth = self.attn(h_rgb16, h_depth16, v16)
        self.last_deltas = {"rgb": d_rgb, "depth": d_depth}
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
        reg_terms = []
        if out["d_rgb"] is not None:
            reg_terms.append(out["d_rgb"].abs().mean())
        if out["d_depth"] is not None:
            reg_terms.append(out["d_depth"].abs().mean())
        l_res = sum(reg_terms) if reg_terms else torch.zeros((), device=x_rgb.device)
        total = l_rgb + l_depth + lambda_res * l_res
        return {"loss": total, "l_rgb16": l_rgb, "l_depth16": l_depth, "l_res": l_res}

    @torch.no_grad()
    def predict(self, x_rgb: torch.Tensor, x_depth: torch.Tensor,
                valid: torch.Tensor | None = None,
                out_size: tuple[int, int] | None = None) -> dict:
        """A_RGB_XMV / A_DEPTH_XMV at input resolution (Phase 2 map protocol)."""
        h, w = x_rgb.shape[-2:]
        out_size = out_size or (h, w)
        fw = self.forward_16(x_rgb, x_depth, valid)
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
