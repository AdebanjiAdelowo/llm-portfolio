"""SAM2 wrapper for interactive / automatic segmentation (stretch goal).

Model: SAM 2 (Meta, 2024)
Install: pip install sam2
Weights: https://github.com/facebookresearch/sam2#model-checkpoints
  - sam2_hiera_large.pt (best quality)
  - sam2_hiera_small.pt (faster)

Usage modes:
  - AUTO: automatic mask generation (no prompts needed)
  - POINT: provide (x, y) foreground point
  - BOX:   provide [x1, y1, x2, y2] bounding box
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

from .base import BaseSegmentationModel


PromptMode = Literal["auto", "point", "box"]


class SAM2Model(BaseSegmentationModel):
    """SAM 2 wrapper supporting automatic and prompted segmentation."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        mode: PromptMode = "auto",
        device: str = "cpu",
    ):
        super().__init__(weights_path, device)
        self.config_path = config_path
        self.mode = mode
        self._predictor = None

    def load(self) -> None:
        # TODO: set up SAM2 predictor
        # from sam2.build_sam import build_sam2
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        #
        # model = build_sam2(self.config_path, self.weights_path, device=self.device)
        # if self.mode == "auto":
        #     self._predictor = SAM2AutomaticMaskGenerator(model)
        # else:
        #     self._predictor = SAM2ImagePredictor(model)
        # self._model = model
        raise NotImplementedError(
            "Download SAM2 weights from https://github.com/facebookresearch/sam2 "
            "and update the import path in this method."
        )

    def predict(
        self,
        image: np.ndarray,
        point: Optional[Tuple[int, int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Run SAM2 inference.

        For 'auto' mode, returns the highest-scoring mask.
        For 'point'/'box' modes, requires the corresponding prompt.
        """
        if self.mode == "auto":
            return self._predict_auto(image)
        elif self.mode == "point" and point is not None:
            return self._predict_point(image, point)
        elif self.mode == "box" and box is not None:
            return self._predict_box(image, box)
        raise ValueError(f"Mode '{self.mode}' requires a matching prompt argument.")

    def _predict_auto(self, image: np.ndarray) -> np.ndarray:
        masks = self._predictor.generate(image)
        if not masks:
            return np.zeros(image.shape[:2], dtype=np.float32)
        best = max(masks, key=lambda m: m["predicted_iou"])
        return best["segmentation"].astype(np.float32)

    def _predict_point(self, image: np.ndarray, point: Tuple[int, int]) -> np.ndarray:
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(np.float32)

    def _predict_box(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            box=np.array(box),
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(np.float32)
