#!/usr/bin/env python3
"""Run inference on a single image or a directory of images.

Usage:
    # Single image
    python scripts/run_inference.py --input product.jpg --model birefnet

    # Batch
    python scripts/run_inference.py --input data/images/ --model birefnet --output outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import list_images
from src.models.birefnet import BiRefNetModel
from src.models.u2net import U2NetModel
from src.inference.pipeline import SegmentationPipeline


MODEL_REGISTRY = {
    "birefnet": lambda args: BiRefNetModel(device=args.device),
    "u2net":    lambda args: U2NetModel(weights_path=args.weights, device=args.device),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run background removal inference.")
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory")
    parser.add_argument("--model", default="birefnet", choices=list(MODEL_REGISTRY))
    parser.add_argument("--output", default="outputs", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--weights", default=None, type=Path, help="Path to model weights")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = MODEL_REGISTRY[args.model](args)
    pipeline = SegmentationPipeline(
        model,
        output_dir=args.output,
        mask_threshold=args.threshold,
        refine=not args.no_refine,
    )

    if args.input.is_dir():
        paths = list_images(args.input)
        print(f"Processing {len(paths)} images with {args.model}...")
        results = pipeline.run_batch(paths)
        print(f"Done. {len(results)} images processed.")
    else:
        result = pipeline.run(args.input)
        print(f"Saved mask + RGBA composite to {args.output}/")
        print(f"Model: {result.model_name}")


if __name__ == "__main__":
    main()
