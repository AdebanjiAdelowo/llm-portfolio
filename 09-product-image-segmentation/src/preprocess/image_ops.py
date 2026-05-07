"""Image preprocessing operations for segmentation and matting models."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Resize image to target_size with letterbox padding (aspect ratio preserved).

    Returns:
        Resized image and a metadata dict needed to undo the transform.
    """
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_value,
    )
    meta = {
        "original_size": (h, w),
        "scale": scale,
        "pad": (pad_top, pad_bottom, pad_left, pad_right),
    }
    return padded, meta


def undo_padding(mask: np.ndarray, meta: dict) -> np.ndarray:
    """Remove letterbox padding and resize mask back to original image dimensions."""
    pad_top, pad_bottom, pad_left, pad_right = meta["pad"]
    h, w = mask.shape[:2]
    cropped = mask[pad_top: h - pad_bottom if pad_bottom else h,
                   pad_left: w - pad_right if pad_right else w]
    orig_h, orig_w = meta["original_size"]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def normalize(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize a uint8 RGB image to float32 using ImageNet stats."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return img


def to_tensor(image: np.ndarray) -> np.ndarray:
    """Convert HxWxC float32 array to CxHxW for PyTorch."""
    return np.transpose(image, (2, 0, 1))


def preprocess_for_inference(
    image: np.ndarray,
    input_size: Tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[np.ndarray, dict]:
    """Full preprocessing pipeline: resize → normalize → CHW tensor.

    Returns:
        (tensor CxHxW float32, pad_meta for undoing resize)
    """
    if image.shape[2] == 4:
        image = image[:, :, :3]
    padded, meta = resize_with_padding(image, input_size)
    normalized = normalize(padded, mean, std)
    tensor = to_tensor(normalized)
    return tensor, meta


def center_crop(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center-crop image to (height, width)."""
    h, w = image.shape[:2]
    th, tw = size
    y = max(0, (h - th) // 2)
    x = max(0, (w - tw) // 2)
    return image[y: y + th, x: x + tw]


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to each channel of an RGB image to improve local contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    channels = cv2.split(image)
    enhanced = [clahe.apply(c) for c in channels]
    return cv2.merge(enhanced)
