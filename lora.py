import math
import torch
from functools import reduce
import os

# adapted from Kohyas LoRA code https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

class LoRAModule(torch.nn.Module):
    def __init__(self, net_name, name, lora_up, lora_down, alpha):
        super().__init__()
        self.net_name = net_name
        self.name = name

        if "unet" in name and ("_proj_" in name or "_conv" in name):
            kernel = lora_down.shape[2]
            padding = 0
            stride = 1

            if kernel == 3:
                padding = 1
            if "downsamplers" in name:
                stride = 2

            self.lora_down = torch.nn.Conv2d(lora_down.shape[1], lora_down.shape[0], (kernel, kernel), (stride, stride), (padding, padding), bias=False)
            self.lora_up = torch.nn.Conv2d(lora_up.shape[1], lora_up.shape[0], (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(lora_down.shape[1], lora_down.shape[0], bias=False)
            self.lora_up = torch.nn.Linear(lora_up.shape[1], lora_up.shape[0], bias=False)
        
        
        self.register_buffer("alpha", torch.tensor(alpha or lora_down.shape[0]))
        self.register_buffer("dim", torch.tensor(lora_down.shape[0]), False)

    def get_weight(self):
        f = (self.alpha / self.dim)
        if type(self.lora_up) == torch.nn.Linear:
            return self.lora_up.weight @ self.lora_down.weight * f
        else:
            down = self.lora_down.weight
            up = self.lora_up.weight
            rank, in_ch, kernel_size, _ = down.shape
            out_ch, _, _, _ = up.shape

            merged = up.reshape(out_ch, -1) @ down.reshape(rank, -1)
            weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)

            return weight * f

    def forward(self, x):
        if type(self.lora_up) == torch.nn.Linear:
            return x @ self.lora_down.weight.T @ self.lora_up.weight.T * (self.alpha / self.dim)
        else:
            return self.lora_up(self.lora_down(x)) * (self.alpha / self.dim)

class LoRANetwork(torch.nn.Module):
    def __init__(self, name, state_dict) -> None:
        super().__init__()
        self.net_name = "lora:" + name.rsplit(".",1)[0].rsplit(os.path.sep,1)[-1]
        self.build_modules(state_dict)
        self.load_state_dict(state_dict, strict=False)

    def build_modules(self, state_dict):

        names = set([k.split(".")[0] for k in state_dict])
        if any([".hada_" in k for k in state_dict]):
            raise RuntimeError("LoHA models are not supported")
        if any([".lokr_" in k for k in state_dict]):
            raise RuntimeError("LoKR models are not supported")
        if any([".mid_" in k for k in state_dict]):
            raise RuntimeError("CP-Decomposition is not supported")

        for name in names:
            up = state_dict[name+".lora_up.weight"]
            down = state_dict[name+".lora_down.weight"]

            alpha = None
            if name+".alpha" in state_dict:
                alpha = state_dict[name+".alpha"].numpy()

            lora = LoRAModule(self.net_name, name, up, down, alpha)
            self.add_module(name, lora)

    def attach(self, models, static):
        for _, module in self.named_modules():
            if not hasattr(module, "name"):
                continue
            name = module.name.replace("lora_", "")

            for model in models:
                if name in model.modules:
                    model.modules[name].attach_lora(module, static)

    def set_strength(self, strength):
        for _, module in self.named_modules():
            if hasattr(module, "multiplier"):
                module.multiplier = torch.tensor(strength).to(self.device)
    
    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        if name == "dtype":
            return next(self.parameters()).dtype
        return super().__getattr__(name)