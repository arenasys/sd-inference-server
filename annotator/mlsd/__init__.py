# MLSD Line Detection
# From https://github.com/navervision/mlsd
# Apache-2.0 license

import cv2
import numpy as np
import torch
import os

from einops import rearrange
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth"

class MLSDdetector:
    def __init__(self, path, download):
        model_path = os.path.join(path, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            download(remote_model_path, model_path)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.cuda().eval()

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype if dtype else self.dtype
        self.model.to(self.device, self.dtype)
        return self

    def __call__(self, input_image, thr_v=0.1, thr_d=0.1):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d, self.device, self.dtype)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass
        return img_output[:, :, 0]
