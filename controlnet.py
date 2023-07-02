import torch
import os
import PIL.Image
import cv2
import einops
import numpy as np
from diffusers.models.controlnet import ControlNetModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import utils

CONTROLNET_MODELS = {
    "Canny": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "Depth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "Pose": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "Lineart": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "Softedge": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",
    "Anime": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth",
    "M-LSD": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth",
    "Instruct": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth",
    "Shuffle": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth",
    "Tile": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
    "Inpaint": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
    "Normal": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
    "Scribble": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "Segmentation": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth"
}

def cv2_to_pil(img):
    return PIL.Image.fromarray(img)

def pil_to_cv2(img):
    return np.array(img)

def annotate(img, annotator, arg):
    img = pil_to_cv2(img)
    if type(annotator) != str:
        img = annotator(img, *arg)
    elif annotator == "invert":
        img = 255 - img  

    c = torch.from_numpy(img).to(torch.float32) / 255.0
    if len(c.shape) == 2:
        c = torch.stack([c]*3)
    if c.shape[-1] == 3:
        c = einops.rearrange(c, "h w c -> c h w")    
    if len(c.shape) == 3:
        c = c.unsqueeze(0)
    return c, cv2_to_pil(img)

def preprocess_control(images, annotators, args, scales):
    conditioning = []
    outputs = []
    for i in range(len(images)):
        s, a, arg, im = scales[i], annotators[i], args[i], images[i]
        cond, out = annotate(im, a, arg)
        outputs += [out]
        conditioning += [(s,cond)]
    return conditioning, outputs

def get_controlnet(name, folder, callback):
    url = CONTROLNET_MODELS[name]
    name = url.rsplit("/",1)[-1]
    file = os.path.join(folder, name)
    if not os.path.exists(file):
        utils.download(url, file, callback)
        return name, True
    return name, False

class ControlledUNET:
    def __init__(self, unet, controlnets):
        self.unet = unet
        self.controlnets = controlnets
        self.controlnet_cond = None
    
    def set_controlnet_conditioning(self, conditioning):
        self.controlnet_cond = [(s,cond.to(self.device, self.dtype)) for s,cond in conditioning]

    def __call__(self, latents, timestep, encoder_hidden_states):
        down_samples, mid_sample = None, None
        for i in range(len(self.controlnets)):
            cn_scale, cn_cond = self.controlnet_cond[i]
            down, mid = self.controlnets[i](
                latents, timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cn_cond,
                conditioning_scale=cn_scale,
                return_dict=False
            )
            if down_samples == None or mid_sample == None:
                down_samples, mid_sample = down, mid
            else:
                for j in range(len(down_samples)):
                    down_samples[j] += down[j]
                mid_sample += mid
            
        return self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_samples, mid_block_additional_residual=mid_sample)
    
    def to(self, *args):
        self.unet.to(*args)
        for c in self.controlnets:
            c.to(*args)

    def __getattr__(self, name):
        return getattr(self.unet, name)