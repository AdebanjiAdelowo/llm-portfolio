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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer


class ProductImageDataset(Dataset):
    """
    Args:
        metadata_csv:   Path to CSV with columns [file_path, caption, split]
        tokenizer:      Primary CLIP tokenizer (CLIP-L for SDXL, or only tokenizer for SD 1.5)
        tokenizer_2:    Secondary tokenizer (OpenCLIP-G for SDXL). None for SD 1.5.
        resolution:     Target image size (square)
        split:          "train" or "val" — filters the CSV
        center_crop:    Apply center crop before resize
        random_flip:    Apply random horizontal flip (train only)
    """

    def __init__(
        self,
        metadata_csv: str,
        tokenizer: CLIPTokenizer,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        resolution: int = 1024,
        split: str = "train",
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution

        self.rows = []
        with open(metadata_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found for split='{split}' in {metadata_csv}")

        transform_list = []
        if center_crop:
            transform_list.append(transforms.CenterCrop(resolution))
        transform_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS))
        if random_flip and split == "train":
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.image_transforms = transforms.Compose(transform_list)

    def _tokenize(self, tokenizer: CLIPTokenizer, caption: str) -> torch.Tensor:
        tokens = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return tokens.input_ids.squeeze(0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        img = Image.open(row["file_path"]).convert("RGB")
        pixel_values = self.image_transforms(img)

        caption = row["caption"]
        item = {
            "pixel_values": pixel_values,
            "input_ids": self._tokenize(self.tokenizer, caption),
            "caption": caption,
            "file_path": row["file_path"],
        }

        if self.tokenizer_2 is not None:
            item["input_ids_2"] = self._tokenize(self.tokenizer_2, caption)

        return item


def build_dataloaders(
    metadata_csv: str,
    tokenizer: CLIPTokenizer,
    tokenizer_2: Optional[CLIPTokenizer] = None,
    resolution: int = 1024,
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    num_workers: int = 2,
) -> tuple:
    """Returns (train_loader, val_loader). val_loader is None if no val split exists."""
    train_ds = ProductImageDataset(
        metadata_csv, tokenizer, tokenizer_2, resolution,
        split="train", center_crop=True, random_flip=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )

    try:
        val_ds = ProductImageDataset(
            metadata_csv, tokenizer, tokenizer_2, resolution,
            split="val", center_crop=True, random_flip=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers,
        )
    except ValueError:
        val_loader = None

    return train_loader, val_loader
