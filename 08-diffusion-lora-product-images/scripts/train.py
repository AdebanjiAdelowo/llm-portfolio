"""
LoRA fine-tuning launcher for SDXL / SD 1.5 on product images.

Run with accelerate for multi-GPU or mixed-precision support:
  accelerate launch scripts/train.py --config configs/lora_config.yaml

Architecture:
  - Loads base diffusion model (SDXL or SD 1.5)
  - Injects PEFT LoRA adapters into U-Net cross-attention
  - Freezes all other weights
  - Trains with standard denoising diffusion loss
  - Validates by generating images at checkpoints
  - Saves LoRA weights (not full model) — much smaller checkpoint
"""

import argparse
import sys
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
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Allow running as `accelerate launch scripts/train.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def encode_prompt_sdxl(batch, text_encoder, text_encoder_2, device, dtype):
    """Encode text with both SDXL text encoders and return concatenated embeddings."""
    # Encoder 1 (CLIP-L): penultimate hidden state, shape [B, 77, 768]
    prompt_embeds_1 = text_encoder(
        batch["input_ids"].to(device),
        output_hidden_states=True,
    ).hidden_states[-2]

    # Encoder 2 (OpenCLIP-G): penultimate hidden state + pooled output
    enc2_out = text_encoder_2(
        batch["input_ids_2"].to(device),
        output_hidden_states=True,
    )
    prompt_embeds_2 = enc2_out.hidden_states[-2]   # [B, 77, 1280]
    pooled_embeds = enc2_out[0]                    # [B, 1280]

    # SDXL U-Net cross-attention expects [B, 77, 2048]
    encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    return encoder_hidden_states.to(dtype), pooled_embeds.to(dtype)


def run_validation(unet, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                   vae, noise_scheduler, cfg, step, accelerator, dtype):
    """Generate validation images and save to disk."""
    val_cfg = cfg["validation"]
    unwrapped_unet = accelerator.unwrap_model(unet)

    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unwrapped_unet,
        scheduler=noise_scheduler,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    images = pipeline(
        prompt=val_cfg["prompt"],
        negative_prompt=val_cfg["negative_prompt"],
        num_images_per_prompt=val_cfg["num_validation_images"],
        num_inference_steps=val_cfg["num_inference_steps"],
        guidance_scale=val_cfg["guidance_scale"],
    ).images

    output_dir = Path(cfg["training"]["output_dir"]) / f"validation_step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(output_dir / f"val_{i:02d}.png")
    print(f"  Validation images saved to {output_dir}")

    del pipeline
    torch.cuda.empty_cache()


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

    # ── Load model components ──────────────────────────────────────────────────
    model_id = cfg["model"]["base_model_id"]
    dtype = torch.float16 if cfg["model"]["torch_dtype"] == "float16" else torch.bfloat16

    # SDXL requires two tokenizers and two text encoders
    tokenizer   = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze everything — only LoRA adapter params will be updated
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet = build_lora_unet(unet, cfg)

    if cfg["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # ── Dataset & Optimizer ────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        metadata_csv=cfg["dataset"]["metadata_file"],
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
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
    device = accelerator.device
    vae.to(device)
    text_encoder.to(device)
    text_encoder_2.to(device)

    # SDXL time conditioning: original size + crop coords + target size (each 2 ints)
    target_size = (cfg["dataset"]["resolution"], cfg["dataset"]["resolution"])

    # ── Training loop ──────────────────────────────────────────────────────────
    global_step = 0
    progress_bar = tqdm(
        total=cfg["training"]["max_train_steps"],
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(cfg["training"]["num_train_epochs"]):
        unet.train()
        for batch in train_loader:
            with accelerator.accumulate(unet):
                bsz = batch["pixel_values"].shape[0]

                # Encode images → latents
                latents = vae.encode(batch["pixel_values"].to(device, dtype=dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timestep
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=device, dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text with both SDXL encoders
                encoder_hidden_states, pooled_embeds = encode_prompt_sdxl(
                    batch, text_encoder, text_encoder_2, device, dtype
                )

                # SDXL added conditioning: time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
                time_ids = torch.tensor(
                    [target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]],
                    dtype=dtype, device=device,
                ).unsqueeze(0).repeat(bsz, 1)

                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}

                # Predict noise
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
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
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                # Validation
                if global_step % cfg["validation"]["validation_steps"] == 0 and accelerator.is_main_process:
                    run_validation(
                        unet, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                        vae, noise_scheduler, cfg, global_step, accelerator, dtype,
                    )
                    unet.train()

                # Checkpoint
                if global_step % cfg["training"]["checkpointing_steps"] == 0:
                    save_path = Path(cfg["training"]["output_dir"]) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    print(f"\nCheckpoint saved: {save_path}")

            if global_step >= cfg["training"]["max_train_steps"]:
                break

        if global_step >= cfg["training"]["max_train_steps"]:
            break

    # ── Save LoRA weights ──────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(cfg["training"]["output_dir"])
        print(f"\nLoRA weights saved to {cfg['training']['output_dir']}")
        print("NEXT STEP: run infer.py to compare base vs. fine-tuned outputs.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
