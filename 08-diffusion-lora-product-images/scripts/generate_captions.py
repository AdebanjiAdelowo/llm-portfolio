"""
BLIP-2 captioning for diffusion LoRA training images.

Generates captions for each image, prepends a trigger word, and writes
a CSV metadata file ready for the training dataset loader.

Usage:
  python scripts/generate_captions.py \
      --image_dir   data/processed/ \
      --output      data/metadata.csv \
      --trigger_word "sks eyewear" \
      --batch_size  4

Requires a GPU. BLIP-2 (2.7B OPT) fits in ~14GB VRAM at fp16.
For <12GB VRAM, use --model Salesforce/blip-image-captioning-base instead.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BLIP-2 captions for training images")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--trigger_word", default="sks eyewear",
                        help="Trigger token prepended to every caption")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b",
                        help="BLIP-2 model ID from Hugging Face Hub")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of images held out for validation")
    return parser.parse_args()


def load_model(model_id: str, device: str):
    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    return processor, model


def caption_batch(images: list, processor, model, device: str, max_new_tokens: int) -> list[str]:
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(images=images, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(ids, skip_special_tokens=True)


def assign_splits(paths: list, val_fraction: float) -> dict:
    import random
    random.seed(42)
    n_val = max(1, int(len(paths) * val_fraction))
    val_set = set(random.sample([str(p) for p in paths], n_val))
    return {str(p): "val" if str(p) in val_set else "train" for p in paths}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CPU inference will be very slow. Use a GPU if possible.")

    image_dir = Path(args.image_dir)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    print(f"Found {len(image_paths)} images in {image_dir}")

    splits = assign_splits(image_paths, args.val_fraction)
    processor, model = load_model(args.model, device)

    rows = []
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        captions = caption_batch(images, processor, model, device, args.max_new_tokens)

        for path, auto_caption in zip(batch_paths, captions):
            auto_caption = auto_caption.strip()
            full_caption = f"a photo of {args.trigger_word} {auto_caption}"
            rows.append({
                "file_path": str(path),
                "caption": full_caption,
                "trigger_word": args.trigger_word,
                "auto_caption": auto_caption,
                "split": splits[str(path)],
            })
            print(f"  {path.name}: {full_caption[:80]}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"\nMetadata written to {output_path}")
    print(f"  Train: {n_train}  |  Val: {n_val}")
    print("\nNEXT STEP: run train.py with --config configs/lora_config.yaml")


if __name__ == "__main__":
    main()
