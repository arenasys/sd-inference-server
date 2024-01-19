# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from einops import rearrange
from .midas.dpt_depth import DPTDepthModel

remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt"

def disabled_train(self, mode=True):
    return self

def load_model(path, download):
    model_path = os.path.join(path, "dpt_hybrid-midas-501f0c75.pt")

    if not os.path.exists(model_path):
        download(remote_model_path, model_path)
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
    )
    return model.eval()

class MiDaSInference(nn.Module):
    def __init__(self, path, download):
        super().__init__()
        model = load_model(path, download)
        self.model = model
        self.model.train = disabled_train

    def forward(self, x):
        with torch.no_grad():
            prediction = self.model(x)
        return prediction

class MidasDetector:
    def __init__(self, path, download):
        self.model = MiDaSInference(path, download)

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype if dtype else self.dtype
        self.model.to(self.device, self.dtype)
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
