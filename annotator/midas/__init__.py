# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference


class MidasDetector:
    def __init__(self, path):
        self.model = MiDaSInference("dpt_hybrid", path)

    def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.model.to(device, dtype)
        return self

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).to(self.device, self.dtype)
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth -= torch.min(depth)
            depth /= torch.max(depth)
            depth = depth.cpu().numpy()
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image
