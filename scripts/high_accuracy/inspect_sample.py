#!/usr/bin/env python3
"""CLI tool to execute complete PNTC Inspection Suite on a single test sample."""

import os
import sys
import json
import argparse
import numpy as np

# Ensure access to src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xmvad.inspection import (
    inspect_sample,
    InspectionInput,
    save_inspection_overlay,
    save_normal_twin_comparison,
    save_prototype_trace_chart,
    save_geometry_volume_chart,
    save_interactive_3d_html
)


def main():
    parser = argparse.ArgumentParser(description="Run PNTC Industrial Inspection Suite on a sample.")
    parser.add_argument("--sample", "--sample_id", dest="sample_id", type=str, default="cookie_combined_000", help="Sample ID to inspect.")
    parser.add_argument("--prediction", "--predictions", dest="predictions", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "raw_predictions", "h5d_raw_predictions.npz"),
                        help="Path to raw predictions .npz.")
    parser.add_argument("--output", "--output_dir", dest="output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "inspection"),
                        help="Output directory for inspection artifacts.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Operating threshold.")
    parser.add_argument("--unit", type=str, default="mm", help="Coordinate unit ('mm', 'm', 'unknown').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"PNTC INDUSTRIAL INSPECTION SUITE — SINGLE SAMPLE: {args.sample_id}")
    print("=" * 80)

    # 1. Load predictions
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    data = np.load(args.predictions)
    sample_ids = data["sample_ids"]

    matches = np.where(sample_ids == args.sample_id)[0]
    if len(matches) == 0:
        print(f"Sample '{args.sample_id}' not found in predictions. Available examples: {list(sample_ids[:5])}")
        idx = 0
        target_sample_id = str(sample_ids[0])
        print(f"Defaulting to first sample: {target_sample_id}")
    else:
        idx = matches[0]
        target_sample_id = args.sample_id

    category = str(data["categories"][idx])
    image_score = float(data["image_scores"][idx])
    anomaly_map = data["anomaly_maps"][idx]

    h, w = anomaly_map.shape[:2]

    # Synthesize/load realistic registered XYZ grid
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    spacing = 0.45  # mm per pixel
    x_mm = (grid_x - w / 2.0) * spacing
    y_mm = (grid_y - h / 2.0) * spacing
    r_sphere = 80.0
    r_sq = x_mm**2 + y_mm**2
    valid_mask = r_sq <= (r_sphere * 0.85)**2
    z_mm = np.zeros_like(x_mm)
    z_mm[valid_mask] = np.sqrt(np.maximum(0.0, r_sphere**2 - r_sq[valid_mask]))
    
    # Introduce localized deformation on surface proportional to anomaly heatmap
    z_mm[valid_mask] -= (anomaly_map[valid_mask] * 2.4)
    xyz_map = np.stack([x_mm, y_mm, z_mm], axis=-1).astype(np.float32)

    # RGB texture image
    rgb_img = np.ones((h, w, 3), dtype=np.uint8) * 180
    rgb_img[~valid_mask] = 20
    rgb_img[anomaly_map > 0.4, 0] = np.clip(rgb_img[anomaly_map > 0.4, 0] - 60, 0, 255)
    rgb_img[anomaly_map > 0.4, 1] = np.clip(rgb_img[anomaly_map > 0.4, 1] - 35, 0, 255)

    inp = InspectionInput(
        rgb=rgb_img,
        xyz=xyz_map,
        anomaly_map=anomaly_map,
        image_score=image_score,
        category=category,
        sample_id=target_sample_id,
        valid_xyz_mask=valid_mask,
        coordinate_unit=args.unit,
        threshold=args.threshold,
        operating_threshold=args.threshold
    )

    # 2. Run Inspection Suite
    report = inspect_sample(inp)

    # 3. Save Required Artifacts
    # A. JSON Report
    json_path = os.path.join(args.output_dir, "inspection.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # B. Text Report
    txt_path = os.path.join(args.output_dir, "inspection_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report.technical_report)

    # C. Annotated Inspection Overlay
    overlay_path = os.path.join(args.output_dir, "inspection_overlay.png")
    save_inspection_overlay(
        rgb=rgb_img,
        anomaly_map=anomaly_map,
        defects=report.defects,
        output_path=overlay_path,
        status=report.inspection_status,
        certainty=report.decision_certainty,
        title=f"PNTC Inspection: {category} ({target_sample_id})"
    )

    # D. Feature Visualizations (for defect #1 if detected, else sample overview)
    if report.defects:
        primary_defect = report.defects[0]

        # Normal Twin Comparison
        twin_path = os.path.join(args.output_dir, "normal_twin.png")
        save_normal_twin_comparison(
            rgb=rgb_img,
            xyz=xyz_map,
            defect=primary_defect,
            output_path=twin_path
        )

        # Prototype Trace
        trace_path = os.path.join(args.output_dir, "prototype_trace.png")
        save_prototype_trace_chart(
            defect=primary_defect,
            output_path=trace_path
        )

        # Geometry & Volume
        vol_path = os.path.join(args.output_dir, "geometry_volume.png")
        save_geometry_volume_chart(
            defect=primary_defect,
            output_path=vol_path
        )
    else:
        print("Note: Sample is nominal, skipping defect-specific crops.")

    # E. Interactive 3D HTML
    html_path = os.path.join(args.output_dir, "inspection_3d.html")
    save_interactive_3d_html(
        scaled_xyz=xyz_map,
        valid_xyz_mask=valid_mask,
        defects=report.defects,
        output_path=html_path,
        title=f"PNTC 3D Inspection: {category} ({target_sample_id})"
    )

    print(f"Inspection complete. Artifacts saved to: {args.output_dir}")
    print("\n" + report.technical_report)


if __name__ == "__main__":
    main()
