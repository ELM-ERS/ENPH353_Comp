# inference_model.py
import os
from typing import Tuple, Optional

import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from model_and_data import CNNRegressor  # reuses your architecture


class CNNInference:
    """
    Wrapper for loading a trained CNNRegressor (.pth) and running fast inference.

    Usage:
        infer = CNNInference("cnn_regressor_680.pth")
        output = infer.predict("some_image.png")
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 680,
        device: Optional[torch.device] = None,
    ):
        """
        model_path: path to .pth file with state_dict
        image_size: expected width/height for the CNN
        device: override device (cpu / cuda). If None → auto-detect.
        """
        self.model_path = model_path
        self.image_size = image_size

        # Pick device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model architecture
        self.model = CNNRegressor(num_outputs=2, input_size=image_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # important!

        # Build same normalization as training
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_image_from_path(self, filename: str) -> Image.Image:
        """Load image using PIL from file path."""
        img = Image.open(filename).convert("RGB")
        return img

    def _load_image_from_cv2(self, img_bgr):
        """Convert cv2 BGR numpy array to a PIL RGB image."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def predict(self, x) -> Tuple[float, float]:
        """
        Input can be:
          - a filename string
          - a cv2 image (numpy array, BGR)

        Returns a Python tuple (x, y)
        """
        # Determine input type
        if isinstance(x, str):
            img = self._load_image_from_path(x)
        else:
            # assume cv2 image array
            img = self._load_image_from_cv2(x)

        # Apply transforms
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        # Inference
        with torch.no_grad():
            output = self.model(tensor)[0]  # shape [2]

        # Convert tensor -> python floats
        return float(output[0]), float(output[1])
