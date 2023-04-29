import os
import torch
from torch import nn
import PIL.Image
import PIL.ImageEnhance
import utils

def relative_file(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

class VAEApprox(nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))
        self.loaded = False

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        return x
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.loaded = True

APPROX_MODEL = VAEApprox()
APPROX_MODEL_PATH = os.path.join("approx", "VAE-approx.pt")

def cheap_preview(latents):
    coefs = torch.tensor([
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]).to(latents.device).to(latents.dtype)
    outputs = torch.zeros([latents.shape[0], 3, latents.shape[2], latents.shape[3]])
    for i in range(latents.shape[0]):
        outputs[i] = torch.einsum("lxy,lr -> rxy", latents[i], coefs).to("cpu")
    outputs = utils.postprocess_images(outputs)
    outputs = [PIL.ImageEnhance.Color(o).enhance(1.5) for o in outputs]
    return outputs

def model_preview(latents):
    if not APPROX_MODEL.loaded:
        APPROX_MODEL.load_state_dict(torch.load(relative_file(APPROX_MODEL_PATH), map_location='cpu'))
    APPROX_MODEL.to(latents.device).to(latents.dtype)
    outputs = utils.postprocess_images(APPROX_MODEL(latents))
    outputs = [PIL.ImageEnhance.Color(o).enhance(1.5) for o in outputs]
    return outputs

def full_preview(latents, vae):
    return utils.postprocess_images(vae.decode(latents.to(vae.dtype)).sample)