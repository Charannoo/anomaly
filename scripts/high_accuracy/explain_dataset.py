#!/usr/bin/env python3
"""CLI tool to generate batch structured explanations across anomalous dataset samples."""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xmvad.explain import explain_anomaly, save_annotated_overlay, ExplainInput


def main():
    parser = argparse.ArgumentParser(description="Generate batch explanations for anomalous samples.")
    parser.add_argument("--predictions", type=str, default=os.path.join(PROJECT_ROOT, "results", "raw_predictions", "h5d_raw_predictions.npz"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "results", "explanations", "batch"))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples_per_category", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PNTC EXPLAIN — BATCH DATASET EXPLANATION GENERATOR")
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
        cat_mask = (categories == cat) & (image_labels == 1)  # Focus on anomalous test samples
        cat_indices = np.where(cat_mask)[0]
        selected_indices = cat_indices[:args.max_samples_per_category]

        print(f"\nProcessing Category: {cat.upper()} ({len(selected_indices)} samples)...")

        for idx in selected_indices:
            s_id = str(sample_ids[idx])
            s_score = float(image_scores[idx])
            s_map = anomaly_maps[idx]

            h, w = s_map.shape[:2]
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            x_mm = (grid_x - w / 2.0) * 0.45
            y_mm = (grid_y - h / 2.0) * 0.45
            valid_mask = (x_mm**2 + y_mm**2) <= (70.0**2)
            z_mm = np.zeros_like(x_mm)
            z_mm[valid_mask] = 40.0 - (s_map[valid_mask] * 2.5)

            xyz_map = np.stack([x_mm, y_mm, z_mm], axis=-1).astype(np.float32)

            rgb_img = np.ones((h, w, 3), dtype=np.uint8) * 175
            rgb_img[~valid_mask] = 25
            rgb_img[s_map > 0.4, 0] = np.clip(rgb_img[s_map > 0.4, 0] - 50, 0, 255)

            inp = ExplainInput(
                rgb=rgb_img,
                xyz=xyz_map,
                anomaly_map=s_map,
                image_score=s_score,
                category=cat,
                sample_id=s_id,
                valid_xyz_mask=valid_mask,
                coordinate_unit="mm",
                threshold=args.threshold
            )

            report = explain_anomaly(inp)
            processed_count += 1

            # Save JSON & overlay
            sample_json_path = os.path.join(args.output_dir, f"{s_id}_explanation.json")
            with open(sample_json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            sample_png_path = os.path.join(args.output_dir, f"{s_id}_overlay.png")
            save_annotated_overlay(
                rgb=rgb_img,
                anomaly_map=s_map,
                defects=report.defects,
                output_path=sample_png_path,
                title=f"PNTC Defect Explanation: {cat} ({s_id})"
            )

            summary_records.append({
                "category": cat,
                "sample_id": s_id,
                "score": round(s_score, 4),
                "num_defects": report.num_defects,
                "summary": report.summary_explanation
            })

    # Save overall dataset summary
    summary_path = os.path.join(args.output_dir, "batch_explanations_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BATCH EXPLANATIONS COMPLETE: Processed {processed_count} samples.")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
