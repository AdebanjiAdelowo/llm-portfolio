"""Abstract base class for all segmentation / matting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np


class BaseSegmentationModel(ABC):
    """Common interface every model wrapper must implement.

    Subclasses handle weight loading, preprocessing quirks, and inference
    details so the pipeline layer can call them interchangeably.
    """

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once before first inference."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single RGB uint8 image.

        Args:
            image: HxWx3 uint8 numpy array in RGB order.

        Returns:
            Float32 alpha/mask array in [0, 1] with shape HxW.
        """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        return self.predict(image)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        loaded = self._model is not None
        return f"{self.name}(device={self.device}, loaded={loaded})"
