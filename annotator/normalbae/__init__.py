# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation
# https://github.com/baegwangbin/surface_normal_uncertainty

import os
import types
import torch
import numpy as np

from einops import rearrange
from .models.NNET import NNET
import torchvision.transforms as transforms

def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

class NormalBaeDetector:
    def __init__(self, path, download):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt"
        model_path = os.path.join(path, "scannet.pt")
        if not os.path.exists(model_path):
            download(remote_model_path, model_path)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        model = model.cuda()
        model.eval()
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype if dtype else self.dtype
        self.model.to(self.device, self.dtype)
        return self

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().cuda()
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal).to(self.device, self.dtype)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            return normal_image
