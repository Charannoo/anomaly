"""RGB appearance deviation analysis and color contrast characterization."""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import ndimage
from .schema import AppearanceInfo


def analyze_appearance(
    rgb: np.ndarray,
    component_mask: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float = 0.5
) -> AppearanceInfo:
    """Analyze visual color and texture deviations of the anomaly region relative to surrounding normal ring.
    
    Args:
        rgb: (H, W, 3) RGB image in uint8 [0, 255] or float [0, 1].
        component_mask: (H, W) boolean mask for anomaly region.
        anomaly_map: (H, W) continuous anomaly heatmap.
        threshold: Operating threshold.
        
    Returns:
        AppearanceInfo instance.
    """
    if rgb.dtype == np.uint8:
        rgb_norm = rgb.astype(np.float32) / 255.0
    else:
        rgb_norm = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    # Surrounding normal ring
    struct = ndimage.generate_binary_structure(2, 2)
    outer = ndimage.binary_dilation(component_mask > 0, structure=struct, iterations=8)
    inner = ndimage.binary_dilation(component_mask > 0, structure=struct, iterations=2)
    ring_mask = outer & (~inner) & (anomaly_map <= threshold * 1.2)

    defect_pixels = rgb_norm[component_mask > 0]
    ring_pixels = rgb_norm[ring_mask] if np.sum(ring_mask) > 10 else rgb_norm[~component_mask]

    if len(defect_pixels) == 0 or len(ring_pixels) == 0:
        return AppearanceInfo(
            deviation_label="minimal",
            appearance_deviation_score=0.0,
            mean_rgb_diff=0.0,
            delta_e_lab=0.0,
            contrast_ratio=1.0,
            texture_variance_diff=0.0
        )

    # Mean RGB difference
    mean_defect_rgb = np.mean(defect_pixels, axis=0)
    mean_ring_rgb = np.mean(ring_pixels, axis=0)
    mean_rgb_diff = float(np.linalg.norm(mean_defect_rgb - mean_ring_rgb))

    # CIE Lab Delta E approximation (standard RGB -> Lab conversion)
    defect_lab = _rgb_to_lab(mean_defect_rgb)
    ring_lab = _rgb_to_lab(mean_ring_rgb)
    delta_e = float(np.linalg.norm(defect_lab - ring_lab))

    # Contrast ratio (luminance difference)
    lum_defect = float(0.2126 * mean_defect_rgb[0] + 0.7152 * mean_defect_rgb[1] + 0.0722 * mean_defect_rgb[2])
    lum_ring = float(0.2126 * mean_ring_rgb[0] + 0.7152 * mean_ring_rgb[1] + 0.0722 * mean_ring_rgb[2])
    contrast_ratio = float((max(lum_defect, lum_ring) + 0.05) / (min(lum_defect, lum_ring) + 0.05))

    # Texture variance
    gray = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
    var_defect = float(np.var(gray[component_mask > 0]))
    var_ring = float(np.var(gray[ring_mask])) if np.sum(ring_mask) > 10 else 0.0
    tex_diff = float(abs(var_defect - var_ring))

    # Normalized score [0, 1]
    # Delta E > 20 is a very noticeable color difference
    score = float(np.clip(0.6 * (delta_e / 25.0) + 0.4 * (mean_rgb_diff / 0.5), 0.0, 1.0))

    if score > 0.65 or delta_e > 18.0:
        dev_label = "strong"
    elif score > 0.30 or delta_e > 8.0:
        dev_label = "moderate"
    else:
        dev_label = "minimal"

    return AppearanceInfo(
        deviation_label=dev_label,
        appearance_deviation_score=round(score, 3),
        mean_rgb_diff=round(mean_rgb_diff, 3),
        delta_e_lab=round(delta_e, 2),
        contrast_ratio=round(contrast_ratio, 2),
        texture_variance_diff=round(tex_diff, 4)
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized RGB [0, 1] to approximate CIE Lab."""
    # Gamma correction to linear sRGB
    def gamma(v):
        return np.where(v > 0.04045, ((v + 0.055) / 1.055) ** 2.4, v / 12.92)

    r, g, b = gamma(rgb[0]), gamma(rgb[1]), gamma(rgb[2])
    
    # sRGB to XYZ (D65 illuminant)
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    def f(t):
        return np.where(t > 0.008856, t ** (1.0 / 3.0), (7.787 * t) + (16.0 / 116.0))

    fx, fy, fz = f(x), f(y), f(z)
    L = max(0.0, min(100.0, (116.0 * fy) - 16.0))
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.array([L, a, b], dtype=np.float32)
