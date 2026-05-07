"""MODNet wrapper for portrait and product matting.

Model: MODNet (Ke et al., 2022)
Weights: https://github.com/ZHKKKe/MODNet
  - modnet_photographic_portrait_matting.ckpt
  - modnet_webcam_portrait_matting.ckpt

Install: pip install modnet  (or clone repo and add to PYTHONPATH)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import BaseSegmentationModel
from ..preprocess.image_ops import preprocess_for_inference, undo_padding


class MODNetModel(BaseSegmentationModel):
    """Wraps MODNet for image matting with soft alpha estimation."""

    INPUT_SIZE = (512, 512)

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(weights_path, device)

    def load(self) -> None:
        # TODO: import MODNet architecture
        # from modnet.models.modnet import MODNet
        # net = MODNet(backbone_pretrained=False)
        # net = torch.nn.DataParallel(net)
        # state = torch.load(self.weights_path, map_location=self.device)
        # net.load_state_dict(state)
        # net.to(self.device).eval()
        # self._model = net
        raise NotImplementedError(
            "Download MODNet weights from https://github.com/ZHKKKe/MODNet "
            "and update the import path in this method."
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        tensor, meta = preprocess_for_inference(image, self.INPUT_SIZE)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # MODNet returns (semantic, detail, matte)
            _, _, matte = self._model(x.float(), True)

        pred = matte.squeeze().cpu().numpy()
        return undo_padding(pred, meta)
