import os
import torch
import numpy as np
from skimage import morphology
from ..lineart import LineartDetector
import cv2
from einops import rearrange

from .ted import TED

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def resize_image_with_pad(img: np.ndarray, resolution: int):
    # Convert greyscale image to RGB.
    if img.ndim == 2:
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)

    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def get_intensity_mask(image_array, lower_bound, upper_bound):
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask

def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result

class AnylineDetector:
    def __init__(self, path, download):
        self.model = self.load_model('MTEED.pth', path, download)
        self.lineart = LineartDetector(path, download)

    def load_model(self, name, path, download):
        remote_model_path = "https://huggingface.co/TheMistoAI/MistoLine/resolve/main/Anyline/" + name
        model_path = os.path.join(path, name)
        if not os.path.exists(model_path):
            download(remote_model_path, model_path)
        model = TED()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model = model.cuda()
        return model
    
    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype if dtype else self.dtype
        self.model.to(self.device, self.dtype)
        self.lineart.to(device, dtype)
        return self
    
    def mteed(self, image, safe_steps=2):
        self.model.to(self.device)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).to(self.device, self.dtype)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge

    def __call__(self, input_image, safe_steps=2):
        assert input_image.ndim == 3
        image = input_image
        
        with torch.no_grad():
            mteed_result = self.mteed(image, safe_steps)
            mteed_result = HWC3(mteed_result)

            lineart_result = self.lineart(image)
            lineart_result = HWC3(lineart_result)
            lineart_result = get_intensity_mask(
                lineart_result, lower_bound=0, upper_bound=1
            )

            cleaned = morphology.remove_small_objects(
                lineart_result.astype(bool), min_size=36, connectivity=1
            )

            lineart_result = lineart_result * cleaned
            final_result = combine_layers(mteed_result, lineart_result)

            return final_result