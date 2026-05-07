"""
LoRA fine-tuning launcher for SDXL / SD 1.5 on product images.

Run with accelerate for multi-GPU or mixed-precision support:
  accelerate launch scripts/train.py --config configs/lora_config.yaml

This scaffold covers the main training loop structure. Adapt the sections
marked [ADAPT] to your specific environment and model choice.

Architecture:
  - Loads base diffusion model (SDXL or SD 1.5)
  - Injects PEFT LoRA adapters into U-Net cross-attention
  - Freezes all other weights
  - Trains with standard denoising diffusion loss
  - Validates by generating images at checkpoints
  - Saves LoRA weights (not full model) — much smaller checkpoint
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from dataset_loader import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to lora_config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lora_unet(unet: UNet2DConditionModel, cfg: dict) -> UNet2DConditionModel:
    """Inject LoRA adapters into U-Net and freeze everything else."""
    lora_cfg = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    return unet


def run_validation(pipeline, cfg: dict, step: int, accelerator: Accelerator):
    """Generate validation images and log to W&B or TensorBoard."""
    val_cfg = cfg["validation"]
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    images = pipeline(
        prompt=val_cfg["prompt"],
        negative_prompt=val_cfg["negative_prompt"],
        num_images_per_prompt=val_cfg["num_validation_images"],
        num_inference_steps=val_cfg["num_inference_steps"],
        guidance_scale=val_cfg["guidance_scale"],
    ).images

    # [ADAPT] Log images to your tracker (W&B, TensorBoard, or just save to disk)
    output_dir = Path(cfg["training"]["output_dir"]) / f"validation_step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(output_dir / f"val_{i:02d}.png")
    print(f"  Validation images saved to {output_dir}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator(
        mixed_precision=cfg["training"]["mixed_precision"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        log_with=cfg["logging"]["report_to"],
        project_dir=cfg["training"]["output_dir"],
    )

    set_seed(cfg["training"]["seed"])

    # ── Load model components ─────────────────────────────────────────────────
    model_id = cfg["model"]["base_model_id"]
    dtype = torch.float16 if cfg["model"]["torch_dtype"] == "float16" else torch.bfloat16

    # [ADAPT] For SD 1.5, replace with single text_encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze everything except LoRA adapters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet = build_lora_unet(unet, cfg)

    if cfg["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # ── Dataset & Optimizer ───────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        metadata_csv=cfg["dataset"]["metadata_file"],
        tokenizer=tokenizer,
        resolution=cfg["dataset"]["resolution"],
        train_batch_size=cfg["training"]["train_batch_size"],
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=cfg["training"]["learning_rate"],
    )

    lr_scheduler = get_scheduler(
        cfg["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=cfg["training"]["lr_warmup_steps"] * cfg["training"]["gradient_accumulation_steps"],
        num_training_steps=cfg["training"]["max_train_steps"] * cfg["training"]["gradient_accumulation_steps"],
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    progress_bar = tqdm(total=cfg["training"]["max_train_steps"], disable=not accelerator.is_local_main_process)

    for epoch in range(cfg["training"]["num_train_epochs"]):
        unet.train()
        for batch in train_loader:
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})

                # [ADAPT] Log to your tracker here
                accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                # Validation
                if global_step % cfg["validation"]["validation_steps"] == 0:
                    # [ADAPT] Build pipeline from current unet and run_validation
                    pass

                # Checkpoint
                if global_step % cfg["training"]["checkpointing_steps"] == 0:
                    save_path = Path(cfg["training"]["output_dir"]) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    print(f"\nCheckpoint saved: {save_path}")

            if global_step >= cfg["training"]["max_train_steps"]:
                break

        if global_step >= cfg["training"]["max_train_steps"]:
            break

    # ── Save LoRA weights ─────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(cfg["training"]["output_dir"])
        print(f"\nLoRA weights saved to {cfg['training']['output_dir']}")
        print("NEXT STEP: run infer.py to compare base vs. fine-tuned outputs.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
