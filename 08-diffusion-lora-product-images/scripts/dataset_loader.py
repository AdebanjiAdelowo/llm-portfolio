"""
PyTorch Dataset for diffusion LoRA fine-tuning.

Reads image paths and captions from the metadata CSV produced by
generate_captions.py and returns pixel tensors + tokenized text.

Used by train.py — import ProductImageDataset from here.
"""

import csv
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


class ProductImageDataset(Dataset):
    """
    Args:
        metadata_csv:   Path to CSV with columns [file_path, caption, split]
        tokenizer:      CLIP tokenizer for text encoding
        resolution:     Target image size (square)
        split:          "train" or "val" — filters the CSV
        center_crop:    Apply center crop before resize
        random_flip:    Apply random horizontal flip (train only)
    """

    def __init__(
        self,
        metadata_csv: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 1024,
        split: str = "train",
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution

        # Load and filter rows for this split
        self.rows = []
        with open(metadata_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found for split='{split}' in {metadata_csv}")

        # Build image transform chain
        transform_list = []
        if center_crop:
            transform_list.append(transforms.CenterCrop(resolution))
        transform_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS))
        if random_flip and split == "train":
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # normalize to [-1, 1]
        ]
        self.image_transforms = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # Load and transform image
        img = Image.open(row["file_path"]).convert("RGB")
        pixel_values = self.image_transforms(img)

        # Tokenize caption
        tokens = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": row["caption"],    # kept as string for logging
            "file_path": row["file_path"],
        }


def build_dataloaders(
    metadata_csv: str,
    tokenizer: CLIPTokenizer,
    resolution: int = 1024,
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    num_workers: int = 2,
) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Convenience function — returns (train_loader, val_loader)."""
    train_ds = ProductImageDataset(
        metadata_csv, tokenizer, resolution, split="train",
        center_crop=True, random_flip=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )

    try:
        val_ds = ProductImageDataset(
            metadata_csv, tokenizer, resolution, split="val",
            center_crop=True, random_flip=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers,
        )
    except ValueError:
        val_loader = None

    return train_loader, val_loader
