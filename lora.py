import math
import torch
from functools import reduce

# adapted from Kohyas LoRA code https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

class LoRAModule(torch.nn.Module):
    def __init__(self, net_name, name, lora_up, lora_down, alpha):
        super().__init__()
        self.net_name = net_name
        self.name = name

        if "unet" in name and "_proj_" in name:
            self.lora_down = torch.nn.Conv2d(lora_down.shape[1], lora_down.shape[0], (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(lora_up.shape[1], lora_up.shape[0], (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(lora_down.shape[1], lora_down.shape[0], bias=False)
            self.lora_up = torch.nn.Linear(lora_up.shape[1], lora_up.shape[0], bias=False)
        
        self.register_buffer("alpha", torch.tensor(alpha or lora_down.shape[0]))
        self.register_buffer("dim", torch.tensor(lora_down.shape[0]), False)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * (self.alpha / self.dim)

class LoRANetwork(torch.nn.Module):
    def __init__(self, name, state_dict) -> None:
        super().__init__()
        self.net_name = name
        self.build_modules(state_dict)
        self.load_state_dict(state_dict, strict=False)

    def build_modules(self, state_dict):
        names = set([k.split(".")[0] for k in state_dict])

        for name in names:
            up = state_dict[name+".lora_up.weight"]
            down = state_dict[name+".lora_down.weight"]

            alpha = None
            if name+".alpha" in state_dict:
                alpha = state_dict[name+".alpha"].numpy()

            lora = LoRAModule(self.net_name, name, up, down, alpha)
            self.add_module(name, lora)

    def attach(self, *models):
        for _, module in self.named_modules():
            if not hasattr(module, "name"):
                continue
            name = module.name.replace("lora_", "")

            for model in models:
                if name in model.modules:
                    model.modules[name].attach_lora(module)

    def set_strength(self, strength):
        for _, module in self.named_modules():
            if hasattr(module, "multiplier"):
                module.multiplier = torch.tensor(strength).to(self.device)
    
    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)