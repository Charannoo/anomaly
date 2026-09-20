"""RGB teacher: frozen pretrained MobileNetV3-Small with intermediate taps.

Centralized feature-tap definition (do NOT scatter integer layer indexes
through the codebase — import ``RETURN_LAYERS`` / ``OUT_CHANNELS`` from here).

Resolution reference (input H x W, verified against torchvision 0.29):

- features[0] Conv2dNormActivation      -> H/2
- features[1] InvertedResidual          -> H/4
- features[2..3] InvertedResidual       -> H/8   (tap ``stage8`` = index 3, 24 ch)
- features[4..8] InvertedResidual       -> H/16  (tap ``stage16`` = index 8, 48 ch)
- features[9..12]                       -> H/32  (unused in Phase 2)
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
    raise ImportError("torchvision is required for the RGB teacher.") from e

# Single source of truth for teacher taps.
RETURN_LAYERS: dict[str, int] = {"stage8": 3, "stage16": 8}
OUT_CHANNELS: dict[str, int] = {"stage8": 24, "stage16": 48}
LEVELS: tuple[str, ...] = ("stage8", "stage16")


class MobilenetBackbone(nn.Module):
    """Run ``features`` and collect tapped intermediate outputs.

    Args:
        features: torchvision ``features`` Sequential (already stem-adapted
            for RGB or depth input).
        return_layers: mapping level name -> block index in ``features``.
    """

    def __init__(self, features: nn.Sequential, return_layers: dict[str, int] | None = None) -> None:
        super().__init__()
        self.features = features
        self.return_layers = dict(return_layers or RETURN_LAYERS)
        self._index_to_name = {idx: name for name, idx in self.return_layers.items()}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self._index_to_name:
                out[self._index_to_name[idx]] = x
        return out


class RGBTeacher(nn.Module):
    """Frozen ImageNet-pretrained MobileNetV3-Small teacher for RGB.

    - Weights frozen (``requires_grad=False``) and kept in eval mode, even
      when the surrounding model is training (see :meth:`train` override).
    - No gradients flow through the teacher (callers additionally run it
      under ``torch.no_grad()``).
    - Forward returns ``{"stage8": ..., "stage16": ...}``.
    """

    return_layers = RETURN_LAYERS
    out_channels = OUT_CHANNELS

    def __init__(self, pretrained: bool = True, return_layers: dict[str, int] | None = None) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            base = mobilenet_v3_small(weights=weights)
        except RuntimeError as e:
            raise RuntimeError(
                "Could not load pretrained MobileNetV3-Small weights "
                f"(pretrained={pretrained}). Check network access / torch cache. "
                f"Original error: {e}"
            ) from e
        self.backbone = MobilenetBackbone(base.features, return_layers or dict(RETURN_LAYERS))
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "RGBTeacher":  # type: ignore[override]
        # Teacher stays frozen in eval mode regardless of outer mode.
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(x)

    @property
    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.parameters())
