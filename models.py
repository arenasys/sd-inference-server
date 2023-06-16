import os
import torch
import utils

from transformers import CLIPTextConfig, CLIPTokenizer
from clip import CustomCLIP
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.models.controlnet import ControlNetModel
from lora import LoRANetwork
from hypernetwork import Hypernetwork

class UNET(UNet2DConditionModel):
    def __init__(self, model_type, model_variant, prediction_type, dtype):
        self.model_type = model_type
        self.inpainting = model_variant == "Inpainting"
        self.prediction_type = prediction_type
        super().__init__(**UNET.get_config(model_type, model_variant))
        self.to(dtype)
        self.additional = None
        
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']
        prediction_type = state_dict['metadata']['prediction_type']
        model_variant = state_dict['metadata'].get('model_variant', "")

        utils.cast_state_dict(state_dict, dtype)
        
        with utils.DisableInitialization():
            unet = UNET(model_type, model_variant, prediction_type, dtype)
            missing, _ = unet.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError("missing keys: " + ", ".join(missing))
        
        unet.additional = AdditionalNetworks(unet)
        return unet

    @staticmethod
    def get_config(model_type, model_variant):
        if model_type == "SDv1":
            config = dict(
                sample_size=32,
                in_channels=9 if model_variant == "Inpainting" else 4,
                out_channels=4,
                down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                cross_attention_dim=768,
                attention_head_dim=8,
            )
        elif model_type == "SDv2":
            config = dict(
                sample_size=32,
                in_channels=9 if model_variant == "Inpainting" else 4,
                out_channels=4,
                down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                cross_attention_dim=1024,
                attention_head_dim=[5, 10, 20, 20],
                use_linear_projection=True
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config

class VAE(AutoencoderKL):
    def __init__(self, model_type, dtype):
        self.model_type = model_type
        super().__init__(**VAE.get_config(model_type))
        self.enable_slicing()
        self.to(dtype)

    class LatentDistribution(DiagonalGaussianDistribution):
        def sample(self, noise):
            x = self.mean + self.std * noise
            return x

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = VAE.LatentDistribution(moments)
        return posterior
        
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']

        utils.cast_state_dict(state_dict, dtype)
        
        with utils.DisableInitialization():
            vae = VAE(model_type, dtype)
            missing, _ = vae.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError("missing keys: " + ", ".join(missing))
        return vae

    @staticmethod
    def get_config(model_type):
        if model_type in {"SDv1", "SDv2"}:
            config = dict(
                sample_size=256,
                in_channels=3,
                out_channels=3,
                down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
                block_out_channels=(128, 256, 512, 512),
                latent_channels=4,
                layers_per_block=2,
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config

class CLIP(CustomCLIP):
    def __init__(self, model_type, dtype):
        self.model_type = model_type
        super().__init__(CLIP.get_config(model_type))
        self.to(dtype)

        self.tokenizer = Tokenizer(model_type)
        self.additional = None
        
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']

        utils.cast_state_dict(state_dict, dtype)
        
        with utils.DisableInitialization():
            clip = CLIP(model_type, dtype)
            missing, _ = clip.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError("missing keys: " + ", ".join(missing))

        clip.additional = AdditionalNetworks(clip)
        return clip

    @staticmethod
    def get_config(model_type):
        if model_type == "SDv1":
            config = CLIPTextConfig(
                attention_dropout=0.0,
                bos_token_id=0,
                dropout=0.0,
                eos_token_id=2,
                hidden_act="quick_gelu",
                hidden_size=768,
                initializer_factor=1.0,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-05,
                max_position_embeddings=77,
                model_type="clip_text_model",
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=1,
                projection_dim=768,
                transformers_version="4.25.1",
                vocab_size=49408
            )
        elif model_type == "SDv2":
            config = CLIPTextConfig(
                vocab_size=49408,
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=23,
                num_attention_heads=16,
                max_position_embeddings=77,
                hidden_act="gelu",
                layer_norm_eps=1e-05,
                dropout=0.0,
                attention_dropout=0.0,
                initializer_range=0.02,
                initializer_factor=1.0,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                model_type="clip_text_model",
                projection_dim=512,
                transformers_version="4.25.0.dev0",
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config

    def set_textual_inversions(self, embeddings):
        tokenized = []
        for name, vec in embeddings.items():
            name = tuple(self.tokenizer(name)["input_ids"][1:-1])
            tokenized += [(name, vec)]
        self.textual_inversions = tokenized

class Tokenizer():
    def __init__(self, model_type):
        tokenizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.model_type = model_type
        self.bos_token_id = 49406
        self.eos_token_id = 49407
        self.pad_token_id = self.eos_token_id if model_type == "SDv1" else 0
        self.comma_token_id = 267
        self.break_token_id = 32472

    def __call__(self, texts):
        return self.tokenizer(texts)

class LoRA(LoRANetwork):
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']

        utils.cast_state_dict(state_dict, dtype)

        model = LoRA("lora:"+name, state_dict)
        model.to(dtype)

        return model

class HN(Hypernetwork):
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']

        utils.cast_state_dict(state_dict, dtype)

        model = HN("hypernet:"+name, state_dict)
        model.to(dtype)

        return model

class AdditionalNetworks():
    class AdditionalModule(torch.nn.Module):
        def __init__(self, parent, name, module: torch.nn.Module):
            super().__init__()
            self.parent = parent
            self.name = name
            self.original_module = module
            self.original_forward = module.forward
            self.dim = module.in_features if hasattr(module, "in_features") else None

            self.hns = []
            self.loras = []

        def forward(self, x):
            if self.hns:
                x = x.clone()
            for hn in self.hns:
                v = hn(x)
                for i in range(len(self.parent.strength)):
                    if hn.net_name in self.parent.strength[i]:
                        x[i] += ((v[i] * self.parent.strength[i][hn.net_name])).clone()
            out = self.original_forward(x)
            for lora in self.loras:
                v = None
                for i in range(len(self.parent.strength)):
                    if lora.net_name in self.parent.strength[i]:
                        if v == None:
                            v = lora(x)
                        out[i] += v[i] * self.parent.strength[i][lora.net_name]
            return out

        def attach_lora(self, module, static):
            if static:
                weight = module.get_weight()
                self.original_module.weight += weight * self.parent.strength[0][module.net_name]
            else:
                self.loras.append(module)
        
        def attach_hn(self, module):
            self.hns.append(module)

        def clear(self):
            self.loras.clear()
            self.hns.clear()

    def __init__(self, model):
        self.modules = {}
        self.strength = {}

        model_type = str(type(model))
        if "CLIP" in model_type:
            self.modules = self.hijack_model(model, 'te', ["CLIPAttention", "CLIPMLP"])
        elif "UNET" in model_type:
            self.modules = self.hijack_model(model, 'unet', ["Transformer2DModel", "Attention"])
        else:
            raise ValueError(f"INVALID TARGET {model_type}")

    def clear(self):
        for name in self.modules:
            self.modules[name].clear()
        self.strength = {}
    
    def set_strength(self, strength):
        self.strength = strength

    def hijack_model(self, model, prefix, targets):
        modules = {}
        for module_name, module in model.named_modules():
            for child_name, child_module in module.named_modules():
                child_class = child_module.__class__.__name__
                if child_class == "Linear" or child_class == "Conv2d":
                    name = (prefix + '.' + module_name + '.' + child_name).replace('.', '_')
                    modules[name] = AdditionalNetworks.AdditionalModule(self, name, child_module)
                    child_module.forward = modules[name].forward
        return modules
    
class ControlNet(ControlNetModel):
    def __init__(self, model_type, dtype):
        self.preprocessor = lambda x: x
        self.model_type = model_type
        super().__init__(**ControlNet.get_config(model_type))
        self.to(dtype)
        
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        utils.cast_state_dict(state_dict, dtype)
        
        with utils.DisableInitialization():
            cn = ControlNet("CN-v1-CANNY", dtype)
            missing, _ = cn.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError("missing keys: " + ", ".join(missing))
        return cn

    @staticmethod
    def get_config(model_type):
        if model_type == "CN-v1-CANNY":
            config = dict(
                cross_attention_dim=768
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config