import torch
import utils

import torchvision.transforms as transforms
import torchvision.transforms.functional
 
from spandrel import ImageModelDescriptor, ModelLoader

TILE_SIZE = 192
TILE_OVERLAP = 8

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
            while True:
                last = image.size[0]
                img = utils.TO_TENSOR(image).unsqueeze(0)
                img = img[:, [2, 1, 0]] # RGB to BGR
                img = img.to(model.device, model.dtype)
                out = upscale_tiled(img, model, TILE_SIZE, TILE_OVERLAP, model.scale)
                out = out.clamp_(0, 1)
                out = out[:, [2, 1, 0]] # BGR to RGB
                image = utils.FROM_TENSOR(out.squeeze(0))
                if last >= image.size[0] or (image.size[0] >= width and image.size[1] >= height):
                    break

            if last >= image.size[0] and (width != image.size[0] or height != image.size[1]):
                raise RuntimeError(f"SR model isnt upscaling ({last} to {image.size[0]})")

            images[i] = upscale_single(image, transforms.InterpolationMode.LANCZOS, width, height)
    
    return images

def upscale_tiled(img, model, tile_size, tile_overlap, scale):
    device = model.device
    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(b, c, h * scale, w * scale, device=device, dtype=img.dtype)
    weights = torch.zeros_like(result)
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size].to(device=device)
            out_patch = model(in_patch)
            result[..., h_idx * scale : (h_idx + tile_size) * scale, w_idx * scale : (w_idx + tile_size) * scale,].add_(out_patch)
            out_patch_mask = torch.ones_like(out_patch)
            weights[..., h_idx * scale : (h_idx + tile_size) * scale, w_idx * scale : (w_idx + tile_size) * scale,].add_(out_patch_mask)

    output = result.div_(weights)

    return output

class SR():
    @staticmethod
    def from_model(name, state_dict, dtype=None):        
        sr = SR()
        sr.model = ModelLoader().load_from_state_dict(state_dict)
        if not isinstance(sr.model, ImageModelDescriptor):
            raise Exception("Not an upscaler")
        sr.model.eval()
        return sr

    def __call__(self, image):
        with torch.no_grad():
            return self.model(image)

    def to(self, device, dtype=None):
        if dtype:
            self.model.model.to(device, dtype)
        else:
            self.model.model.to(device)
        return self

    def __getattr__(self, name):
        if name == "device":
            return next(self.model.model.parameters()).device
        if name == "dtype":
            return next(self.model.model.parameters()).dtype
        return getattr(self.model, name)