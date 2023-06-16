import torch
import PIL.Image
import cv2
import einops
import numpy as np
from diffusers.models.controlnet import ControlNetModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def cv2_to_pil(img):
    return PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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