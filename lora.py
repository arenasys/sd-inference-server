import math
import torch
from functools import reduce
import os
import tqdm

# adapted from Kohyas LoRA code https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

class LoRAModule(torch.nn.Module):
    def __init__(self, net_name, layer_name, shape, dim, alpha, kernel=None, stride=None, padding=None):
        super().__init__()
        self.net_name = net_name
        self.layer_name = layer_name

        if kernel != None:
            self.lora_down = torch.nn.Conv2d(shape[1], dim, (kernel, kernel), (stride, stride), (padding, padding), bias=False)
            self.lora_up = torch.nn.Conv2d(dim, shape[0], (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(shape[1], dim, bias=False)
            self.lora_up = torch.nn.Linear(dim, shape[0], bias=False)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("dim", torch.tensor(dim), False)

    def from_weights(net_name, layer_name, lora_up, lora_down, alpha):
        shape = (lora_up.shape[0], lora_down.shape[1])
        dim = lora_down.shape[0]

        if len(lora_down.shape) > 2:
            kernel = lora_down.shape[2]
            padding = 0
            stride = 1

            if kernel == 3:
                padding = 1
            if "downsamplers" in layer_name or "op" in layer_name:
                stride = 2

            return LoRAModule(net_name, layer_name, shape, dim, alpha, kernel, stride, padding)
        else:
            return LoRAModule(net_name, layer_name, shape, dim, alpha)
        
    def from_module(net_name, layer_name, module, dim, alpha):
        if module.__class__.__name__ == "Conv2d":
            in_dim = module.in_channels
            out_dim = module.out_channels
            kernel = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            return LoRAModule(net_name, layer_name, (out_dim, in_dim), dim, alpha, kernel, stride, padding)
        else:
            in_dim = module.in_features
            out_dim = module.out_features
            return LoRAModule(net_name, layer_name, (out_dim, in_dim), dim, alpha)

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
    
    def to(self, *args):
        self.lora_down = self.lora_down.to(*args)
        self.lora_up = self.lora_up.to(*args)
        return self

class LoRANetwork(torch.nn.Module):
    def __init__(self, net_name) -> None:
        super().__init__()
        self.net_name = net_name
        self.decomposition = {}

    def from_state_dict(self, state_dict):
        self.build_modules(state_dict)
        self.load_state_dict(state_dict, strict=False)

    def from_modules(self, modules, dim, alpha, conv=True):
        for name, module in modules.items():
            is_conv = "resnet" in name or "sample" in name
            if is_conv and not conv:
                continue
            lora = LoRAModule.from_module(self.net_name, name, module, dim, alpha)
            self.add_module(name, lora)

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

            lora = LoRAModule.from_weights(self.net_name, name, up, down, alpha)
            self.add_module(name, lora)

    def compose(self):
        state_dict = {}
        self.to(torch.device("cpu"), torch.float32)
        for _, module in self.named_modules():
            if not hasattr(module, "layer_name"):
                continue
            state_dict[module.layer_name] = module.get_weight().to(torch.float16)
        self.to(torch.device("cpu"), torch.float16)
        return state_dict
    
    def precompute_decomposition(self, device, callback=None):
        if self.decomposition:
            return

        state_dict = self.compose()

        iter = tqdm.tqdm(state_dict)
        for k in iter:
            if callback:
                callback(iter.format_dict)

            mat = state_dict[k].float()
            size = mat.size()

            conv2d = (len(size) == 4)
            kernel_size = None if not conv2d else size[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            if max(mat.shape) > 4096 or torch.get_num_threads() < 6:
                mat = mat.to(device)
            
            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :256].to("cpu").contiguous()
            S = S[:256].to("cpu").contiguous()
            Vh = Vh[:256, :].to("cpu").contiguous()

            self.decomposition[k] = (U,S,Vh,size)

    def get_key_at_rank(decomposition, rank, conv_rank):
        U, S, Vh, size = decomposition

        conv2d = (len(size) == 4)
        kernel_size = None if not conv2d else size[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        out_dim, in_dim = size[0:2]

        module_new_rank = conv_rank if conv2d_3x3 else rank
        module_new_rank = min(module_new_rank, in_dim, out_dim)

        U = U[:, :module_new_rank].float()
        S = S[:module_new_rank].float()
        U = U @ torch.diag(S)

        Vh = Vh[:module_new_rank, :].float()

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, 0.99)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
            U = U.reshape(out_dim, module_new_rank, 1, 1)
            Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

        up = U.to("cpu").contiguous().half()
        down = Vh.to("cpu").contiguous().half()
        alpha = torch.tensor(module_new_rank).half()

        return up, down, alpha

    def attach(self, model, static):
        if static:
            if self.net_name in model.static:
                return
            model.static[self.net_name] = model.get_strength(0, self.net_name)

        for _, module in self.named_modules():
            if type(module) != LoRAModule:
                continue
            name = module.layer_name.replace("lora_", "")
            if name in model.modules:
                model.modules[name].attach_lora(module, static)

    def set_strength(self, strength):
        for _, module in self.named_modules():
            if hasattr(module, "multiplier"):
                module.multiplier = torch.tensor(strength).to(self.device)
    
    def to(self, *args):
        for _, module in self.named_modules():
            if type(module) in {LoRAModule}:
                module.to(*args)
        return self

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        if name == "dtype":
            return next(self.parameters()).dtype
        return super().__getattr__(name)