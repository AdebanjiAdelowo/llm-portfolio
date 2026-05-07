#!/usr/bin/env python3
"""Evaluate one or more models against ground-truth masks.

Usage:
    python scripts/evaluate.py \
        --image-dir data/images \
        --mask-dir  data/masks \
        --models    birefnet u2net \
        --output    outputs/eval/results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.io import collect_pairs, load_image, load_mask
from src.evaluation.metrics import compute_all_metrics, average_metrics
from src.models.birefnet import BiRefNetModel
from src.models.u2net import U2NetModel
from src.inference.pipeline import SegmentationPipeline


MODEL_REGISTRY = {
    "birefnet": lambda: BiRefNetModel(device="cpu"),
    "u2net":    lambda: U2NetModel(device="cpu"),
}


def build_results_table(all_results: dict[str, dict]) -> str:
    """Format a Markdown table for README embedding."""
    metrics = ["iou", "dice", "mad", "boundary_f1"]
    header = "| Model | IoU ↑ | Dice ↑ | MAD ↓ | BoundaryF1 ↑ | Avg Speed (ms/img) |"
    sep    = "|-------|-------|--------|-------|--------------|-------------------|"
    rows = [header, sep]
    for model_name, data in all_results.items():
        avg = data["avg_metrics"]
        spd = data.get("avg_ms", -1)
        row = (
            f"| {model_name} "
            f"| {avg.get('iou', 0):.3f} "
            f"| {avg.get('dice', 0):.3f} "
            f"| {avg.get('mad', 0):.4f} "
            f"| {avg.get('boundary_f1', 0):.3f} "
            f"| {spd:.0f} |"
        )
        rows.append(row)
    return "\n".join(rows)


def run_evaluation(args: argparse.Namespace) -> None:
    pairs = collect_pairs(args.image_dir, args.mask_dir)
    if not pairs:
        raise RuntimeError(f"No matched image/mask pairs in {args.image_dir} / {args.mask_dir}")

    print(f"Found {len(pairs)} image/mask pairs.")
    all_results: dict[str, dict] = {}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model '{model_name}', skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        model = MODEL_REGISTRY[model_name]()
        pipeline = SegmentationPipeline(model, output_dir=args.output_dir, refine=args.refine)

        per_image_metrics = []
        timings = []

        for img_path, mask_path in pairs:
            image = load_image(img_path, mode="RGB")
            gt_mask = load_mask(mask_path)

            t0 = time.perf_counter()
            result = pipeline.run(image, save_outputs=args.save_outputs, stem=img_path.stem)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)

            metrics = compute_all_metrics(result.refined_mask, gt_mask)
            per_image_metrics.append(metrics)

            print(
                f"  {img_path.name:30s} | "
                f"IoU={metrics.iou:.3f} | "
                f"Dice={metrics.dice:.3f} | "
                f"MAD={metrics.mad:.4f} | "
                f"{elapsed_ms:.0f}ms"
            )

        avg = average_metrics(per_image_metrics)
        all_results[model_name] = {
            "avg_metrics": avg,
            "avg_ms": float(np.mean(timings)),
        }
        print(f"\n  Average IoU: {avg['iou']:.3f}  |  Average MAD: {avg['mad']:.4f}")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print Markdown table
    print("\n--- Markdown Table (paste into README) ---")
    print(build_results_table(all_results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation models.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["birefnet"], choices=list(MODEL_REGISTRY))
    parser.add_argument("--output", default="outputs/eval/results.json", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--refine", action="store_true", default=True)
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
