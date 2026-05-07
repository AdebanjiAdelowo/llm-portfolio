"""U2Net wrapper for salient object detection / background removal.

Model: U-2-Net (Qin et al., 2020)
Weights: https://github.com/xuebinqin/U-2-Net
  - u2net.pth       (~176 MB) — full model, best quality
  - u2netp.pth      (~4 MB)   — lightweight, faster

Install: pip install u2net  (or clone the repo and add to PYTHONPATH)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseSegmentationModel
from ..preprocess.image_ops import preprocess_for_inference, undo_padding


class U2NetModel(BaseSegmentationModel):
    """Wraps U²-Net for single-image salient object detection."""

    INPUT_SIZE = (320, 320)

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_variant: str = "u2net",
        device: str = "cpu",
    ):
        super().__init__(weights_path, device)
        self.model_variant = model_variant  # "u2net" or "u2netp"

    def load(self) -> None:
        # TODO: import U2NET architecture from u2net package or local clone
        # from u2net import U2NET, U2NETP
        # arch = U2NET(3, 1) if self.model_variant == "u2net" else U2NETP(3, 1)
        # state = torch.load(self.weights_path, map_location=self.device)
        # arch.load_state_dict(state)
        # arch.to(self.device).eval()
        # self._model = arch
        raise NotImplementedError(
            "Download u2net.pth from https://github.com/xuebinqin/U-2-Net "
            "and update the import path in this method."
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        tensor, meta = preprocess_for_inference(image, self.INPUT_SIZE)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # 1xCxHxW

        with torch.no_grad():
            # U2Net returns 7 side outputs; d0 is the final fused prediction
            d0, *_ = self._model(x.float())

        pred = d0.squeeze().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return undo_padding(pred, meta)
