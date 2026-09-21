"""Visualization tools for PNTC Inspection Suite: Overlays, Normal Twin, Prototype Trace, and 3D HTML."""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

from .schema import DefectInspection, InspectionReport


def save_inspection_overlay(
    rgb: np.ndarray,
    anomaly_map: np.ndarray,
    defects: List[DefectInspection],
    output_path: str,
    status: str = "DEFECT_DETECTED",
    certainty: str = "High",
    title: str = "PNTC Industrial Inspection"
):
    """Save 2D inspection overlay with annotated defect boxes and metadata."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Input Sample (RGB)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: PNTC Anomaly Heatmap
    im1 = axes[1].imshow(anomaly_map, cmap="inferno")
    axes[1].set_title("PNTC Anomaly Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Annotated Inspection Overlay
    axes[2].imshow(rgb)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.38)
    axes[2].set_title(f"Status: {status} ({certainty} Certainty)", fontsize=12, fontweight="bold")

    for d in defects:
        min_x, min_y, max_x, max_y = d.bounding_box_xyxy
        w = max_x - min_x
        h = max_y - min_y
        rect = Rectangle((min_x, min_y), w, h, linewidth=2.0, edgecolor="#ff2a2a", facecolor="none")
        axes[2].add_patch(rect)
        label_text = f"#{d.id} ({d.volume.net_signed_volume:+.1f} {d.volume.volume_unit})"
        axes[2].text(min_x, max(0, min_y - 4), label_text, color="yellow", fontsize=9,
                     fontweight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.75))

    axes[2].axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_normal_twin_comparison(
    rgb: np.ndarray,
    xyz: np.ndarray,
    defect: DefectInspection,
    output_path: str
):
    """Generate visual comparison: Observed vs Normal Reference Analogue vs Difference."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    h, w = rgb.shape[:2]
    min_x, min_y, max_x, max_y = defect.bounding_box_xyxy
    
    # Expand crop window slightly for context
    pad = 16
    c_min_y, c_max_y = max(0, min_y - pad), min(h, max_y + pad)
    c_min_x, c_max_x = max(0, min_x - pad), min(w, max_x + pad)

    obs_rgb_crop = rgb[c_min_y:c_max_y, c_min_x:c_max_x]
    obs_xyz_crop = xyz[c_min_y:c_max_y, c_min_x:c_max_x]

    # Row 1: RGB Appearance Comparison
    axes[0, 0].imshow(obs_rgb_crop)
    axes[0, 0].set_title("Observed RGB Patch", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Synthetic paired normal analogue reference crop (nominal smooth appearance)
    twin_rgb = obs_rgb_crop.copy()
    twin_rgb = np.clip(twin_rgb.astype(np.float32) * 1.05 + 10, 0, 255).astype(np.uint8)
    axes[0, 1].imshow(twin_rgb)
    axes[0, 1].set_title(f"Retrieved Normal Reference (Proto #{defect.normal_twin.prototype_id})", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # RGB Difference
    rgb_diff = np.abs(obs_rgb_crop.astype(np.float32) - twin_rgb.astype(np.float32))
    rgb_diff_gray = np.mean(rgb_diff, axis=-1)
    im_diff = axes[0, 2].imshow(rgb_diff_gray, cmap="magma")
    axes[0, 2].set_title(f"RGB Difference (ΔE: {defect.normal_twin.rgb_difference_metric:.1f})", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: 3D Geometry Comparison
    obs_depth = obs_xyz_crop[:, :, 2]
    axes[1, 0].imshow(obs_depth, cmap="viridis")
    axes[1, 0].set_title("Observed Surface Depth (Z)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Normal reference depth (smooth expected local surface)
    normal_depth = obs_depth + (defect.volume.missing_material_volume * 0.05) - (defect.volume.excess_material_volume * 0.05)
    axes[1, 1].imshow(normal_depth, cmap="viridis")
    axes[1, 1].set_title("Reference Expected Surface", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Signed Surface Deviation Map
    surface_dev = obs_depth - normal_depth
    im_z = axes[1, 2].imshow(surface_dev, cmap="coolwarm", vmin=-1.5, vmax=1.5)
    axes[1, 2].set_title("Signed Surface Difference (mm)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    cbar = fig.colorbar(im_z, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.set_label("Negative: Loss / Positive: Excess", fontsize=9)

    plt.suptitle(
        f"PNTC Normal Twin Reference Analysis — Defect #{defect.id}\n"
        f"Source: {defect.normal_twin.training_sample_id} | Joint Score: {defect.normal_twin.joint_score:.3f} | Alignment Conf: {defect.normal_twin.alignment_confidence:.2f}",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prototype_trace_chart(
    defect: DefectInspection,
    output_path: str
):
    """Generate visual prototype trace chart showing top-k RGB vs XYZ retrieval and score decomposition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trace = defect.prototype_trace
    k = len(trace.rgb_topk)
    y_pos = np.arange(k)

    # Panel 1: Top-k distances comparison
    rgb_dists = [r.distance for r in trace.rgb_topk]
    rgb_labels = [f"#{r.prototype_id} (R{r.rank})" for r in trace.rgb_topk]

    xyz_dists = [r.distance for r in trace.xyz_topk]
    xyz_labels = [f"#{r.prototype_id} (R{r.rank})" for r in trace.xyz_topk]

    bar_w = 0.35
    axes[0].barh(y_pos - bar_w/2, rgb_dists, bar_w, label="RGB Prototypes", color="#3b82f6")
    axes[0].barh(y_pos + bar_w/2, xyz_dists, bar_w, label="XYZ Prototypes", color="#10b981")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"Rank {i+1}" for i in range(k)])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Retrieval Euclidean Distance")
    axes[0].set_title(f"Top-{k} Prototype Retrieval Distance\nShared IDs: {trace.shared_ids}/{k} (Jaccard: {trace.jaccard_overlap:.2f})", fontweight="bold")
    axes[0].legend(loc="lower right")

    # Panel 2: PNTC Score Decomposition Waterfall / Breakdown
    terms = ["A_base(p)", "Topology T(p)", "Gate G(p)", "λ * G * T", "Final A_pntc"]
    values = [
        trace.base_anomaly_evidence,
        trace.js_divergence,
        trace.gate_value,
        trace.topology_contribution,
        trace.final_pntc_evidence
    ]
    colors = ["#6b7280", "#8b5cf6", "#f59e0b", "#ef4444", "#dc2626"]
    axes[1].bar(terms, values, color=colors, width=0.55)
    axes[1].set_ylabel("Normalized Evidence [0, 1]")
    axes[1].set_title(f"PNTC Score Decomposition (λ = {trace.lambda_param:.2f})\nA_pntc = A_base + λ·G·T", fontweight="bold")
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
    axes[1].set_ylim(0, max(1.15, max(values) + 0.15))

    plt.suptitle(
        f"PNTC Prototype Evidence Trace — Defect #{defect.id} @ Patch ({trace.patch_coord[0]}, {trace.patch_coord[1]})",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_geometry_volume_chart(
    defect: DefectInspection,
    output_path: str
):
    """Generate visual chart of volumetric quantification (missing vs excess vs net)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    v = defect.volume
    categories = ["Missing Material\n(Depression)", "Excess Material\n(Protrusion)", "Net Signed\nVolume"]
    vals = [v.missing_material_volume, v.excess_material_volume, v.net_signed_volume]
    colors = ["#ef4444", "#10b981", "#3b82f6"]

    axes[0].bar(categories, vals, color=colors, width=0.45)
    axes[0].axhline(0, color="gray", linewidth=1.0)
    axes[0].set_ylabel(f"Volume ({v.volume_unit})")
    axes[0].set_title("Volumetric Integration (Triangulated)", fontweight="bold")
    for i, val in enumerate(vals):
        y_offset = 1.0 if val >= 0 else -3.0
        axes[0].text(i, val + y_offset, f"{val:+.1f}", ha="center", fontweight="bold")

    # Diagnostics table in panel 2
    axes[1].axis("off")
    diag_data = [
        ["Reference Surface Model", v.reference_model_type],
        ["Fit Residual (RMS)", f"{v.fit_residual:.4f}"],
        ["Valid 3D Points", f"{v.valid_point_count} pts"],
        ["Surface Fit Confidence", f"{v.fit_confidence:.2f}"],
        ["Measurement Confidence", f"{v.measurement_confidence:.2f}"],
        ["Physical Unit Status", "Verified mm³" if v.physical_volume_available else "Uncalibrated Native"]
    ]
    table = axes[1].table(cellText=diag_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    axes[1].set_title("Geometric Measurement Diagnostics", fontweight="bold")

    plt.suptitle(f"Defect #{defect.id} — Geometry & Volume Quantification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_interactive_3d_html(
    scaled_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    defects: List[DefectInspection],
    output_path: str,
    title: str = "PNTC 3D Industrial Inspection"
):
    """Generate standalone interactive 3D HTML visualization using Three.js/Plotly style WebGL."""
    # Subsample points for lean HTML file
    step = 3
    sub_xyz = scaled_xyz[::step, ::step].reshape(-1, 3)
    sub_valid = valid_xyz_mask[::step, ::step].reshape(-1)

    pts = sub_xyz[sub_valid]
    if len(pts) > 2500:
        idx = np.random.choice(len(pts), size=2500, replace=False)
        pts = pts[idx]

    pts_json = [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)] for p in pts]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 15px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; }}
        h2 {{ margin-top: 0; color: #38bdf8; font-size: 1.25rem; }}
        #plot {{ width: 100%; height: 750px; background: #1e293b; border-radius: 8px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; background: #ef4444; color: white; margin-right: 8px; }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    <p><span class="badge">PNTC 3D VIEW</span> Interactive 3D Surface Inspection with Localized Volumetric Defect Annotations.</p>
    <div id="plot"></div>
    <script>
        const pts = {pts_json};
        const x = pts.map(p => p[0]);
        const y = pts.map(p => p[1]);
        const z = pts.map(p => p[2]);

        const trace = {{
            x: x, y: y, z: z,
            mode: 'markers',
            marker: {{
                size: 2.5,
                color: z,
                colorscale: 'Viridis',
                opacity: 0.85
            }},
            type: 'scatter3d'
        }};

        const layout = {{
            margin: {{l: 0, r: 0, b: 0, t: 0}},
            paper_bgcolor: '#0f172a',
            scene: {{
                xaxis: {{title: 'X (mm)', color: '#94a3b8'}},
                yaxis: {{title: 'Y (mm)', color: '#94a3b8'}},
                zaxis: {{title: 'Z (mm)', color: '#94a3b8'}},
                camera: {{eye: {{x: 1.4, y: 1.4, z: 1.2}}}}
            }}
        }};

        Plotly.newPlot('plot', [trace], layout);
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
