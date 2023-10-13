import torch
import os
import PIL.Image
import einops
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import utils
from annotator import shuffle, canny

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
    "Segmentation": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "QR": "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/blob/main/control_v1p_sd15_qrcode_monster.safetensors"
}

def cv2_to_pil(img):
    return PIL.Image.fromarray(img)

def pil_to_cv2(img):
    return np.array(img)

def annotate(img, annotator, model, arg, mask=None):
    img = pil_to_cv2(img)
    if annotator == "Invert":
        img = 255 - img
    elif annotator == "Canny":
        img = canny(img, *arg)
    elif annotator == "Shuffle":
        img = shuffle(img)
    elif annotator == "Pose":
        img, pose = model(img, *arg)
    elif model:
        img = model(img, *arg)

    c = torch.from_numpy(img).to(torch.float32) / 255.0

    if len(c.shape) == 2:
        c = torch.stack([c]*3)
    if c.shape[-1] == 3:
        c = einops.rearrange(c, "h w c -> c h w")    
    if len(c.shape) == 3:
        c = c.unsqueeze(0)

    if annotator == "Inpaint" and mask != None:
        mask = utils.TO_TENSOR(mask).to(torch.float32)
        mask = torch.cat((mask, mask, mask), 0).unsqueeze(0)
        c[mask > 0.5] = -1
        img_mask = einops.rearrange(mask[0], "c h w -> h w c").numpy()
        img[img_mask > 0.5] = 0

    return c, cv2_to_pil(img)

def preprocess_control(images, annotators, models, args, scales, masks=[]):
    conditioning = []
    outputs = []

    if masks and any(masks):
        mask = [m for m in masks if m][0]
    else:
        mask = None

    for i in range(len(images)):
        sc, an, md, arg, im = scales[i], annotators[i], models[i], args[i], images[i]
        cond, out = annotate(im, an, md, arg, mask)
        outputs += [out]
        conditioning += [(sc,cond)]
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
    
    def set_controlnet_conditioning(self, conditioning, device):
        self.controlnet_cond = [(s,cond.to(device, self.dtype)) for s,cond in conditioning]

    def __call__(self, latents, timestep, encoder_hidden_states, **kwargs):
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
            
        return self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_samples, mid_block_additional_residual=mid_sample, **kwargs)
    
    def to(self, *args):
        self.unet.to(*args)
        for c in self.controlnets:
            c.to(*args)

    def __getattr__(self, name):
        return getattr(self.unet, name)