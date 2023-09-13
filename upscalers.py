import torch
import utils

import torchvision.transforms as transforms
import torchvision.transforms.functional
 
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_single(input, mode, width, height):    
    if type(input) == torch.Tensor:
        rw = width / input.shape[-1]
        rh = height / input.shape[-2]
    else:
        rw = width / input.size[0]
        rh = height / input.size[1]

    if abs(rw-rh) < 0.01:
        z = min(width, height)
    elif rw > rh:
        z = width
    else:
        z = height
    
    resize = transforms.transforms.Resize(z, mode)
    input = resize(input)

    if type(input) == torch.Tensor:
        dx = int((input.shape[-1]-width)*0.5)
        dy = int((input.shape[-2]-height)*0.5)
    else:
        dx = int((input.size[0]-width)*0.5)
        dy = int((input.size[1]-height)*0.5)

    input = torchvision.transforms.functional.crop(input, dy, dx, height, width)
    return input

def upscale(inputs, mode, width, height):
    return [upscale_single(inputs[i], mode, width, height) for i in range(len(inputs))]

def upscale_super_resolution(images, model, width, height):
    images = [i for i in images]

    with torch.inference_mode():
        for i, image in enumerate(images):
            last = -1
            while last < image.size[0] and (image.size[0] < width or image.size[1] < height):
                last = image.size[0]
                img = utils.TO_TENSOR(image).unsqueeze(0)
                img = img.to(model.device, model.dtype)
                out = model(img)
                out = out.clamp_(0, 1)
                image = utils.FROM_TENSOR(out.squeeze(0))
            if last >= image.size[0]:
                raise RuntimeError(f"SR model isnt upscaling ({last} to {image.size[0]})")

            images[i] = upscale_single(image, transforms.InterpolationMode.LANCZOS, width, height)
    
    return images

class SR(RRDBNet):
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if len(state_dict) == 1:
            state_dict = next(iter(state_dict.values()))

        if "model.0.weight" in state_dict:
            state_dict = SR.convert(state_dict)

        if "RRDB_trunk.0.RDB1.conv1.weight" in state_dict:
            state_dict = SR.BSRGANConvert(state_dict)
        
        num_block = max([int(k.split(".")[1]) for k in state_dict if k.startswith("body.")] + [0]) + 1

        if num_block == 0:
            raise ValueError(f"unknown upscaler format")
        
        model = SR(3, 3, 4, 64, num_block)
        model.load_state_dict(state_dict)

        return model
    
    @staticmethod
    def convert(state_dict):
        for k in list(state_dict.keys()):
            kk = k.lower()
            kk = kk.replace("model.1.sub.", "body.")
            kk = kk.replace(".0.weight", ".weight")
            kk = kk.replace(".0.bias", ".bias")
            if k != kk:
                state_dict[kk] = state_dict[k]
                del state_dict[k]

        KEYS = {
            "model.": "conv_first.",
            "model.3.": "conv_up1.",
            "model.6.": "conv_up2.",
            "model.8.": "conv_hr.",
            "body.23.": "conv_body.",
            "model.10.": "conv_last.",
        }

        for k in KEYS:
            for s in ["weight", "bias"]:
                state_dict[KEYS[k]+s] = state_dict[k+s]
                del state_dict[k+s]
        
        return state_dict
    
    @staticmethod
    def BSRGANConvert(state_dict):
        KEYS = {
            "rrdb_trunk.": "body.",
            "trunk_conv.": "conv_body.",
            "upconv1.": "conv_up1.",
            "upconv2.": "conv_up2.",
            "hrconv.": "conv_hr."
        }
        
        for k in list(state_dict.keys()):
            kk = k.lower()
            for s in KEYS:
                if kk.startswith(s):
                    kk = kk.replace(s, KEYS[s])
            if kk != k:
                state_dict[kk] = state_dict[k]
                del state_dict[k]
        
        return state_dict

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        if name == "dtype":
            return next(self.parameters()).dtype
        return super().__getattr__(name)