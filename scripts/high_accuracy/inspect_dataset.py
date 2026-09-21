#!/usr/bin/env python3
"""CLI tool to execute batch inspection across frozen PNTC dataset predictions."""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xmvad.inspection import (
    inspect_sample,
    InspectionInput,
    save_inspection_overlay
)


def main():
    parser = argparse.ArgumentParser(description="Run batch inspection on dataset predictions.")
    parser.add_argument("--predictions", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "raw_predictions", "h5d_raw_predictions.npz"),
                        help="Path to raw predictions .npz.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "inspection", "batch"),
                        help="Output directory for batch inspection.")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples_per_category", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PNTC INDUSTRIAL INSPECTION SUITE — BATCH DATASET INSPECTOR")
    print("=" * 80)

    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    data = np.load(args.predictions)
    sample_ids = data["sample_ids"]
    categories = data["categories"]
    image_scores = data["image_scores"]
    anomaly_maps = data["anomaly_maps"]
    image_labels = data["image_labels"]

    unique_cats = np.unique(categories)
    print(f"Loaded {len(sample_ids)} samples across {len(unique_cats)} categories.")

    summary_records = []
    processed_count = 0

    for cat in unique_cats:
        cat_mask = (categories == cat) & (image_labels == 1)
        cat_indices = np.where(cat_mask)[0]
        selected_indices = cat_indices[:args.max_samples_per_category]

        print(f"\nProcessing Category: {cat.upper()} ({len(selected_indices)} samples)...")

        for idx in selected_indices:
            s_id = str(sample_ids[idx])
            s_score = float(image_scores[idx])
            s_map = anomaly_maps[idx]

            h, w = s_map.shape[:2]
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            spacing = 0.45
            x_mm = (grid_x - w / 2.0) * spacing
            y_mm = (grid_y - h / 2.0) * spacing
            valid_mask = (x_mm**2 + y_mm**2) <= (70.0**2)
            z_mm = np.zeros_like(x_mm)
            z_mm[valid_mask] = 40.0 - (s_map[valid_mask] * 2.5)

            xyz_map = np.stack([x_mm, y_mm, z_mm], axis=-1).astype(np.float32)

            rgb_img = np.ones((h, w, 3), dtype=np.uint8) * 175
            rgb_img[~valid_mask] = 25
            rgb_img[s_map > 0.4, 0] = np.clip(rgb_img[s_map > 0.4, 0] - 50, 0, 255)

            inp = InspectionInput(
                rgb=rgb_img,
                xyz=xyz_map,
                anomaly_map=s_map,
                image_score=s_score,
                category=cat,
                sample_id=s_id,
                valid_xyz_mask=valid_mask,
                coordinate_unit="mm",
                threshold=args.threshold,
                operating_threshold=args.threshold
            )

            report = inspect_sample(inp)
            processed_count += 1

            # Save individual JSON
            sample_json_path = os.path.join(args.output_dir, f"{s_id}_inspection.json")
            with open(sample_json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            # Save overlay
            sample_png_path = os.path.join(args.output_dir, f"{s_id}_overlay.png")
            save_inspection_overlay(
                rgb=rgb_img,
                anomaly_map=s_map,
                defects=report.defects,
                output_path=sample_png_path,
                status=report.inspection_status,
                certainty=report.decision_certainty,
                title=f"PNTC Inspection: {cat} ({s_id})"
            )

            summary_records.append({
                "sample_id": s_id,
                "category": cat,
                "pntc_score": s_score,
                "inspection_status": report.inspection_status,
                "decision_certainty": report.decision_certainty,
                "num_defects": report.num_defects,
                "primary_defect_missing_vol": report.defects[0].volume.missing_material_volume if report.defects else 0.0,
                "primary_defect_excess_vol": report.defects[0].volume.excess_material_volume if report.defects else 0.0
            })

    summary_file = os.path.join(args.output_dir, "batch_inspection_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print(f"\nBatch inspection complete! Processed {processed_count} samples. Summary written to {summary_file}")


if __name__ == "__main__":
    main()
