import torch
import lycoris
import utils
lycoris.logger.disabled = True

# lycoris_lora workarounds

class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, lora_layer = None, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, hidden_states, scale = 1.0):
        return super().forward(hidden_states)
    
class Linear(torch.nn.Linear):
    def __init__(self, *args, lora_layer = None, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, hidden_states, scale = 1.0):
        return super().forward(hidden_states)

import diffusers.models.lora
diffusers.models.lora.LoRACompatibleConv = Conv2d
diffusers.models.lora.LoRACompatibleLinear = Linear

locon_forward = lycoris.LoConModule.forward
def new_locon_forward(self, x, *args, **kwargs): return locon_forward(self, x)
lycoris.LoConModule.forward = new_locon_forward

loha_forward = lycoris.LohaModule.forward
def new_loha_forward(self, x, *args, **kwargs): return loha_forward(self, x)
lycoris.LohaModule.forward = new_loha_forward

lokr_forward = lycoris.LokrModule.forward
def new_lokr_forward(self, x, *args, **kwargs): return lokr_forward(self, x)
lycoris.LokrModule.forward = new_lokr_forward

full_forward = lycoris.FullModule.forward
def new_full_forward(self, x, *args, **kwargs): return full_forward(self, x)
lycoris.FullModule.forward = new_full_forward

norm_forward = lycoris.NormModule.forward
def new_norm_forward(self, x, *args, **kwargs): return norm_forward(self, x)
lycoris.NormModule.forward = new_norm_forward

class LycorisNetwork():
    def __init__(self, net_name="") -> None:
        super().__init__()
        self.net_name = net_name
        self.network: lycoris.kohya.LycorisNetworkKohya = None
        self.weights = None
        self.strength = 1.0

    def from_state_dict(self, state_dict):
        self.weights = state_dict

    def build_network(self, unet, clip):
        weights = self.weights
        if unet.model_type == "SDXL-Base":
            weights = {utils.lora_mapping(k):v for k,v in self.weights.items()}

        self.network, _ = lycoris.kohya.create_network_from_weights(
            1.0, None, None, clip, unet, weights
        )
        self.detach_unet()
        self.detach_text_encoder()

    def get_loras(self):
        return self.network.unet_loras + self.network.text_encoder_loras

    def attach_unet(self):
        for lora in self.network.unet_loras:
            lora.apply_to()

    def attach_text_encoder(self):
        for lora in self.network.text_encoder_loras:
            lora.apply_to()

    def detach_unet(self):
        for lora in self.network.unet_loras:
            lora.restore()

    def detach_text_encoder(self):
        for lora in self.network.text_encoder_loras:
            lora.restore()

    def merge_unet(self):
        for lora in self.network.unet_loras:
            lora.merge_to(self.strength)

    def merge_text_encoder(self):
        for lora in self.network.text_encoder_loras:
            lora.merge_to(self.strength)

    def set_strength(self, strength):
        self.strength = strength
        for lora in self.get_loras():
            lora.multiplier = self.strength
    
    def to(self, *args):
        if self.weights:
            for k in self.weights:
                self.weights[k].to(*args)
                self.weights[k].requires_grad = False
        if self.network:
            for lora in self.get_loras():
                lora.to(*args)
                lora.requires_grad = False
        return self
    
    def get_device(self):
        for lora in self.get_loras():
            return next(lora.parameters()).device