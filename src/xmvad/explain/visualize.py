"""Visualization generation for 2D annotated anomaly overlays and 3D interactive views."""

from typing import List, Optional
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .schema import DefectRegion, ExplanationReport


def save_annotated_overlay(
    rgb: np.ndarray,
    anomaly_map: np.ndarray,
    defects: List[DefectRegion],
    output_path: str,
    title: Optional[str] = None
) -> str:
    """Render and save publication-quality 2D RGB + Anomaly heatmap overlay with defect annotations.
    
    Args:
        rgb: (H, W, 3) RGB image in uint8 or float [0, 1].
        anomaly_map: (H, W) continuous anomaly heatmap.
        defects: List of DefectRegion instances.
        output_path: Output PNG file path.
        title: Optional figure title.
        
    Returns:
        output_path: Path to saved image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if rgb.dtype != np.uint8:
        rgb_disp = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        rgb_disp = rgb.copy()

    h, w = rgb_disp.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    # Base RGB image
    ax.imshow(rgb_disp)

    # Colormapped anomaly heatmap with alpha transparency
    cmap = plt.get_cmap('jet')
    norm_map = np.clip(anomaly_map, 0.0, 1.0)
    heatmap_colored = cmap(norm_map)
    # Set transparency based on anomaly score: low scores are transparent
    alpha_mask = np.clip((norm_map - 0.2) / 0.8, 0.0, 0.65)
    heatmap_colored[:, :, 3] = alpha_mask
    ax.imshow(heatmap_colored)

    # Draw defect contours, bounding boxes, and label tags
    colors = ['#FF0055', '#00FFCC', '#FFCC00', '#CC00FF', '#0099FF']
    for idx, d in enumerate(defects):
        col = colors[idx % len(colors)]
        min_x, min_y, max_x, max_y = d.bounding_box_xyxy
        bw = max_x - min_x
        bh = max_y - min_y

        # Bounding box
        rect = patches.Rectangle(
            (min_x, min_y), bw, bh,
            linewidth=2.0, edgecolor=col, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)

        # Centroid marker
        cx, cy = d.location.centroid_px
        ax.plot(cx, cy, marker='+', markersize=8, markeredgewidth=2, color=col)

        # Annotation label
        if d.size.physical_units_available and d.size.major_length_mm is not None:
            size_txt = f"{d.size.major_length_mm}×{d.size.minor_length_mm}mm"
        else:
            size_txt = f"{d.size.area_px}px"

        struct_txt = d.geometry.label
        if d.geometry.max_depression is not None and d.geometry.max_depression > 0.05:
            struct_txt += f" ({d.geometry.max_depression}{d.geometry.unit})"
        elif d.geometry.max_protrusion is not None and d.geometry.max_protrusion > 0.05:
            struct_txt += f" (+{d.geometry.max_protrusion}{d.geometry.unit})"

        label_txt = f"#{d.id}: {d.shape.label}\n{size_txt} | {struct_txt}"
        
        # Position label above or below box
        tag_y = max(12, min_y - 8)
        ax.text(
            min_x, tag_y, label_txt,
            color='white', fontsize=7.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=col, edgecolor='none', alpha=0.85)
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    return output_path


def save_interactive_3d_html(
    scaled_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    defects: List[DefectRegion],
    output_path: str,
    title: str = "3D Point Cloud Defect Visualization"
) -> str:
    """Generate a lightweight interactive HTML 3D point cloud viewer."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Subsample points for smooth browser rendering (max 15,000 points)
    valid_indices = np.argwhere(valid_xyz_mask)
    if len(valid_indices) > 15000:
        step = len(valid_indices) // 15000 + 1
        valid_indices = valid_indices[::step]

    pts = []
    labels = []
    colors = []

    # Assign labels and colors
    defect_mask_combined = np.zeros(valid_xyz_mask.shape, dtype=int)
    for d in defects:
        defect_mask_combined[d.mask > 0] = d.id

    for y, x in valid_indices:
        p = scaled_xyz[y, x]
        d_id = defect_mask_combined[y, x]
        pts.append(p)
        if d_id > 0:
            labels.append(f"Defect #{d_id}")
            colors.append("#FF0055")
        else:
            labels.append("Normal Surface")
            colors.append("#3399FF")

    pts_arr = np.array(pts)
    
    # Generate standalone Plotly-based HTML if plotly is available, else create standalone Canvas3D HTML
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Scatter3d(
            x=pts_arr[:, 0],
            y=pts_arr[:, 1],
            z=pts_arr[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,
                color=colors,
                opacity=0.8
            ),
            text=labels,
            hoverinfo='text'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            margin=dict(r=0, l=0, b=0, t=40)
        )
        fig.write_html(output_path)
    except Exception:
        # Fallback simple HTML template with pre-baked JSON data
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="margin:0; background-color:#111;">
<div id="plot3d" style="width:100vw; height:100vh;"></div>
<script>
var data = [{{
  x: {pts_arr[:, 0].tolist()},
  y: {pts_arr[:, 1].tolist()},
  z: {pts_arr[:, 2].tolist()},
  mode: 'markers',
  marker: {{ size: 2.5, color: {colors}, opacity: 0.8 }},
  text: {labels},
  type: 'scatter3d'
}}];
var layout = {{
  title: {{ text: '{title}', font: {{ color: '#fff' }} }},
  paper_bgcolor: '#111',
  scene: {{
    xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333' }},
    yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333' }},
    zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333' }},
    aspectmode: 'data'
  }}
}};
Plotly.newPlot('plot3d', data, layout);
</script>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return output_path
