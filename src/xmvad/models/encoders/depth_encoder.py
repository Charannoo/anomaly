"""Depth teacher: separate MobileNetV3-Small for 1-channel Z-depth.

Initialization (documented, per research plan):
- Start from ImageNet-pretrained RGB weights.
- Adapt ONLY the stem convolution from 3ch -> 1ch by averaging the RGB
  kernels over the channel dimension::

      W_depth[:, 0, :, :] = mean(W_rgb, dim=channel)

  This preserves the pretrained filter structure (edge-like filters stay
  edge-like) while accepting single-channel input.
- All remaining layers keep their pretrained weights unchanged.
- The teacher is then FROZEN (same policy as the RGB teacher).
- Weights are NEVER shared with the RGB teacher (independent module).

A later ablation (pretrained-derived vs random init) can load this class
with ``pretrained=False``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        mobilenet_v3_small,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("torchvision is required for the depth teacher.") from e

from xmvad.models.encoders.rgb_encoder import LEVELS, OUT_CHANNELS, RETURN_LAYERS, MobilenetBackbone


def adapt_stem_to_1ch(rgb_weight: torch.Tensor) -> torch.Tensor:
    """Average a ``[C_out, 3, kH, kW]`` stem kernel to ``[C_out, 1, kH, kW]``."""
    if rgb_weight.dim() != 4 or rgb_weight.shape[1] != 3:
        raise ValueError(f"Expected [C,3,kH,kW] RGB stem weight, got {tuple(rgb_weight.shape)}")
    return rgb_weight.mean(dim=1, keepdim=True)


class DepthTeacher(nn.Module):
    """Frozen depth teacher operating on normalized 1-channel Z input."""

    return_layers = RETURN_LAYERS
    out_channels = OUT_CHANNELS
    levels = LEVELS

    def __init__(self, pretrained: bool = True, return_layers: dict[str, int] | None = None) -> None:
        super().__init__()
        rgb_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            rgb_base = mobilenet_v3_small(weights=rgb_weights)
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not load pretrained weights for depth init: {e}"
            ) from e
        depth_base = mobilenet_v3_small(weights=None)
        # The depth stem is ALWAYS 1-channel (random init by default).
        stem: nn.Conv2d = depth_base.features[0][0]
        depth_base.features[0][0] = nn.Conv2d(
            1, stem.out_channels, kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        if pretrained:
            state = rgb_base.state_dict()
            stem_w = state["features.0.0.weight"]  # [16, 3, 3, 3]
            state["features.0.0.weight"] = adapt_stem_to_1ch(stem_w)
            missing = depth_base.load_state_dict(state, strict=False)
            # Shapes now match — assert cleanliness explicitly.
            if missing.missing_keys or missing.unexpected_keys:  # pragma: no cover
                raise RuntimeError(f"Depth weight load mismatch: {missing}")
        self.backbone = MobilenetBackbone(depth_base.features, return_layers or dict(RETURN_LAYERS))
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "DepthTeacher":  # type: ignore[override]
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1] != 1:
            raise ValueError(f"Depth teacher expects 1-channel input, got {x.shape[1]} channels")
        return self.backbone(x)

    @property
    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.parameters())

    @property
    def stem_in_channels(self) -> int:
        conv = self.backbone.features[0][0]
        return int(conv.in_channels)
