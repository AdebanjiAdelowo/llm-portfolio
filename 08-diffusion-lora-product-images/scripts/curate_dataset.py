"""
Dataset curation pipeline for diffusion LoRA fine-tuning on product images.

Pipeline stages:
  1. Load raw images from --input directory
  2. Filter by minimum resolution and aspect ratio
  3. Resize / center-crop to target resolution
  4. Remove near-duplicates via perceptual hashing
  5. Export cleaned images to --output directory

Usage:
  python scripts/curate_dataset.py \
      --input  data/raw/ \
      --output data/processed/ \
      --resolution 1024 \
      --min_resolution 512 \
      --max_aspect_ratio 1.5

Adapt thresholds to your dataset before running.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from PIL import Image

# imagehash is optional but strongly recommended for dedup
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("WARNING: imagehash not installed — duplicate detection disabled.")
    print("         Run: pip install imagehash")


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate product image dataset")
    parser.add_argument("--input", required=True, help="Directory of raw images")
    parser.add_argument("--output", required=True, help="Directory for processed images")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Target output resolution (square)")
    parser.add_argument("--min_resolution", type=int, default=512,
                        help="Minimum acceptable side length of source image")
    parser.add_argument("--max_aspect_ratio", type=float, default=1.5,
                        help="Maximum width/height or height/width ratio to keep")
    parser.add_argument("--hash_threshold", type=int, default=8,
                        help="Max pHash distance to flag as duplicate (lower = stricter)")
    return parser.parse_args()


def meets_resolution_requirement(img: Image.Image, min_res: int) -> bool:
    return min(img.width, img.height) >= min_res


def meets_aspect_ratio_requirement(img: Image.Image, max_ratio: float) -> bool:
    ratio = max(img.width, img.height) / min(img.width, img.height)
    return ratio <= max_ratio


def center_crop_and_resize(img: Image.Image, size: int) -> Image.Image:
    """Crop to square from center, then resize to target."""
    w, h = img.width, img.height
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((size, size), Image.LANCZOS)


def compute_phash(img: Image.Image):
    if not IMAGEHASH_AVAILABLE:
        return None
    return imagehash.phash(img)


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"Found {len(image_paths)} candidate images in {input_dir}")

    seen_hashes = {}
    stats = {"total": len(image_paths), "kept": 0, "filtered_res": 0,
             "filtered_aspect": 0, "filtered_dup": 0, "errors": 0}

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  ERROR opening {path.name}: {e}")
            stats["errors"] += 1
            continue

        if not meets_resolution_requirement(img, args.min_resolution):
            print(f"  SKIP (low res {img.width}×{img.height}): {path.name}")
            stats["filtered_res"] += 1
            continue

        if not meets_aspect_ratio_requirement(img, args.max_aspect_ratio):
            print(f"  SKIP (aspect ratio {img.width/img.height:.2f}): {path.name}")
            stats["filtered_aspect"] += 1
            continue

        phash = compute_phash(img)
        if phash is not None:
            is_dup = any(
                abs(phash - h) <= args.hash_threshold
                for h in seen_hashes
            )
            if is_dup:
                print(f"  SKIP (duplicate): {path.name}")
                stats["filtered_dup"] += 1
                continue
            seen_hashes[phash] = path.name

        processed = center_crop_and_resize(img, args.resolution)

        out_path = output_dir / path.name
        processed.save(out_path, quality=95)
        stats["kept"] += 1
        print(f"  OK: {path.name}")

    print("\n── Curation Summary ──────────────────────────")
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    print(f"\nProcessed images saved to: {output_dir}")
    print("\nNEXT STEP: run generate_captions.py on the output directory.")


if __name__ == "__main__":
    main()
