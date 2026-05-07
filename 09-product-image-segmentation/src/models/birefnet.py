"""BiRefNet wrapper — SOTA transformer-based background removal.

Model: BiRefNet (Zheng et al., 2024)
HuggingFace: zhengpeng7/BiRefNet
  pip install transformers torch

This is the easiest model to get running because weights are on HuggingFace.
No separate weight download needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import BaseSegmentationModel
from ..preprocess.image_ops import resize_with_padding, normalize, to_tensor, undo_padding


class BiRefNetModel(BaseSegmentationModel):
    """Wraps BiRefNet via HuggingFace transformers for zero-shot background removal."""

    INPUT_SIZE = (1024, 1024)

    def __init__(self, device: str = "cpu"):
        super().__init__(weights_path=None, device=device)
        self._processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageSegmentation
        from torchvision.transforms.functional import normalize as tv_normalize

        self._model = AutoModelForImageSegmentation.from_pretrained(
            "zhengpeng7/BiRefNet", trust_remote_code=True, torch_dtype=torch.float32
        )
        self._model.to(self.device).eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 4:
            image = image[:, :, :3]

        padded, meta = resize_with_padding(image, self.INPUT_SIZE)
        tensor = to_tensor(normalize(padded)).astype(np.float32)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self._model(x)[-1].sigmoid()

        pred = preds.squeeze().cpu().numpy()
        return undo_padding(pred, meta)
