"""
Inference script: compare base model vs. LoRA fine-tuned model.

Generates a side-by-side grid for a set of test prompts and saves it to disk.

Usage:
  python scripts/infer.py \
      --base_model stabilityai/stable-diffusion-xl-base-1.0 \
      --lora_weights outputs/lora_weights/ \
      --prompts_file configs/eval_prompts.txt \
      --output results/comparison_grid.png \
      --num_images 4

outputs/lora_weights/ should be the directory produced by train.py.
"""

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image


DEFAULT_PROMPTS = [
    "a product photo of sks eyewear on white background, studio lighting, sharp focus",
    "a product photo of sks eyewear, side profile, minimal background",
    "sks eyewear displayed on a clean surface, editorial photography",
    "close-up of sks eyewear temple detail, macro photography",
]

NEGATIVE_PROMPT = "blurry, low quality, distorted, cartoon, painting, illustration, bokeh, grainy"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate base vs. LoRA comparison grid")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_weights", required=True, help="Directory with saved LoRA weights")
    parser.add_argument("--prompts_file", default=None, help="Text file with one prompt per line")
    parser.add_argument("--output", default="results/comparison_grid.png")
    parser.add_argument("--num_images", type=int, default=4, help="Images per prompt per model")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompts(prompts_file) -> list:
    if prompts_file and Path(prompts_file).exists():
        return [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    return DEFAULT_PROMPTS


def load_base_pipeline(model_id: str) -> DiffusionPipeline:
    # Try fp16 variant first (official SDXL release); fall back to default weights
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    except Exception:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True,
        )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_images(pipe, prompts: list[str], num_images: int, steps: int,
                    guidance_scale: float, seed: int) -> list[Image.Image]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    all_images = []
    for prompt in prompts:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        all_images.extend(result.images)
    return all_images


def make_comparison_grid(base_imgs: list[Image.Image], lora_imgs: list[Image.Image],
                          prompts: list[str], num_per_prompt: int) -> Image.Image:
    """
    Lays out a grid: [BASE row | LORA row] for each prompt.
    Prompts run vertically; base/lora run horizontally in pairs.
    """
    assert len(base_imgs) == len(lora_imgs)
    thumb = 256
    cols = num_per_prompt * 2   # base columns + lora columns
    rows = len(prompts)
    grid = Image.new("RGB", (cols * thumb, rows * thumb), color=(240, 240, 240))

    for prompt_idx in range(rows):
        for img_idx in range(num_per_prompt):
            # Base image
            base = base_imgs[prompt_idx * num_per_prompt + img_idx].resize((thumb, thumb))
            grid.paste(base, (img_idx * thumb, prompt_idx * thumb))
            # LoRA image
            lora = lora_imgs[prompt_idx * num_per_prompt + img_idx].resize((thumb, thumb))
            grid.paste(lora, ((num_per_prompt + img_idx) * thumb, prompt_idx * thumb))

    return grid


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    print(f"Running inference on {len(prompts)} prompts × {args.num_images} images × 2 models")

    # ── Base model ──────────────────────────────────────────────────────────
    print("\n[1/2] Generating base model images...")
    base_pipe = load_base_pipeline(args.base_model)
    base_images = generate_images(base_pipe, prompts, args.num_images, args.steps,
                                  args.guidance_scale, args.seed)
    del base_pipe
    torch.cuda.empty_cache()

    # ── LoRA fine-tuned model ───────────────────────────────────────────────
    print("[2/2] Generating LoRA fine-tuned images...")
    lora_pipe = load_base_pipeline(args.base_model)
    lora_pipe.load_lora_weights(args.lora_weights)
    lora_images = generate_images(lora_pipe, prompts, args.num_images, args.steps,
                                  args.guidance_scale, args.seed)
    del lora_pipe
    torch.cuda.empty_cache()

    # ── Compose and save grid ───────────────────────────────────────────────
    grid = make_comparison_grid(base_images, lora_images, prompts, args.num_images)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"\nComparison grid saved to {output_path}")
    print("Left columns = base model | Right columns = LoRA fine-tuned")

    # Also save individual images for results writeup
    for i, img in enumerate(base_images):
        img.save(output_path.parent / f"base_{i:03d}.png")
    for i, img in enumerate(lora_images):
        img.save(output_path.parent / f"lora_{i:03d}.png")
    print(f"Individual images saved to {output_path.parent}/")


if __name__ == "__main__":
    main()
