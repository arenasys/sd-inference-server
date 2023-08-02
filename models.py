import os
import torch
import utils

from transformers import CLIPTextConfig, CLIPTokenizer
from clip import CustomCLIP, CustomSDXLCLIP
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.models.controlnet import ControlNetModel
from lora import LoRANetwork
from hypernetwork import Hypernetwork

class UNET(UNet2DConditionModel):
    def __init__(self, model_type, model_variant, prediction_type, dtype):
        self.model_type = model_type
        self.model_variant = model_variant
        self.inpainting = model_variant == "Inpainting"
        self.prediction_type = prediction_type
        self.upcast_attention = True if model_variant == "SDv2.1" else False
        self.determined = False

        super().__init__(**UNET.get_config(model_type, model_variant))
        self.to(dtype)
        self.additional = None

    def __call__(self, *args, **kwargs):
        if not 'added_cond_kwargs' in kwargs:
            kwargs['added_cond_kwargs'] = {}
        kwargs['cross_attention_kwargs'] = {'upcast_attention': self.upcast_attention}
        return super().__call__(*args, **kwargs)
        
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
            raise ValueError("missing keys in UNET: " + ", ".join(missing))
        
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
        elif model_type == "SDXL-Base":
            config = dict(
                sample_size=128,
                in_channels=4,
                out_channels=4,
                down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'),
                up_block_types=('CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D'),
                block_out_channels=(320, 640, 1280),
                cross_attention_dim=2048,
                transformer_layers_per_block=[1, 2, 10],
                attention_head_dim=[5, 10, 20],
                use_linear_projection=True,
                addition_embed_type='text_time',
                addition_time_embed_dim=256,
                projection_class_embeddings_input_dim=2816
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config
    
    def determine_type(self):
        needs_type = self.prediction_type == "unknown" or self.model_type == "SDv2"
        if self.determined or not needs_type:
            return
        self.determined = True

        test_cond = torch.ones((1, 2, self.config.cross_attention_dim), device=self.device, dtype=self.dtype) * 0.5
        test_latent = torch.ones((1, self.config.in_channels, 8, 8), device=self.device, dtype=self.dtype) * 0.5
        test_timestep = torch.asarray([999], device=self.device, dtype=self.dtype)

        test_pred = self(test_latent, test_timestep, encoder_hidden_states=test_cond).sample
        if torch.isnan(test_pred).any():
            print('UPCASTING ATTENTION')
            self.upcast_attention = True
            self.model_variant = "SDv2.1"
            test_pred = self(test_latent, test_timestep, encoder_hidden_states=test_cond).sample

        is_v = (test_pred - 0.5).mean().item() < -1
        self.prediction_type = "v" if is_v else "epsilon"
        print("USING",self.prediction_type,"PREDICTION")

class VAE(AutoencoderKL):
    def __init__(self, model_type, dtype):
        self.model_type = model_type
        config = VAE.get_config(model_type)
        super().__init__(**config)
        self.scaling_factor = config["scaling_factor"]
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
            raise ValueError("missing keys in VAE: " + ", ".join(missing))
        return vae

    @staticmethod
    def get_config(model_type):
        config = dict(
            sample_size=256,
            in_channels=3,
            out_channels=3,
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(128, 256, 512, 512),
            latent_channels=4,
            layers_per_block=2,
            scaling_factor=0.18215
        )
        if model_type == "SDv1":
            config["sample_size"] = 512
        elif model_type == "SDv2":
            config["sample_size"] = 768
        elif model_type == "SDXL-Base":
            config["sample_size"] = 1024
            config["scaling_factor"] = 0.13025
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config

class CLIP(torch.nn.Module):
    def __init__(self, model_type, dtype):
        super().__init__()
        self.model_type = model_type
        if model_type in {"SDv1", "SDv2"}:
            self.model = CustomCLIP(CLIP.get_config(model_type))
        elif model_type == "SDXL-Base":
            self.model = CustomSDXLCLIP(CLIP.get_config(model_type, "OpenCLIP"), CLIP.get_config(model_type, "LDM"))
        self.to(dtype)
        self.tokenizer = Tokenizer(model_type)
        self.additional = None

    def encode(self, input_ids, clip_skip):
        return self.model(input_ids, clip_skip)
    
    def state_dict(self):
        return self.model.state_dict()

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)

    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']

        utils.cast_state_dict(state_dict, dtype)
        
        with utils.DisableInitialization():
            clip = CLIP(model_type, dtype)
            missing, _ = clip.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError("missing keys in CLIP: " + ", ".join(missing))

        clip.additional = AdditionalNetworks(clip.model)
        return clip

    @staticmethod
    def get_config(model_type, model_variant=""):
        if model_type == "SDv1" or (model_type == "SDXL-Base" and model_variant == "LDM"):
            config = CLIPTextConfig(
                hidden_act="quick_gelu",
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=12,
                projection_dim=768,
            )
        elif model_type == "SDv2":
            config = CLIPTextConfig(
                hidden_act="gelu",
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=23,
                projection_dim=512
            )
        elif model_type == "SDXL-Base" and model_variant == "OpenCLIP":
            config = CLIPTextConfig(
                hidden_act="gelu",
                hidden_size=1280,
                intermediate_size=5120,
                num_attention_heads=20,
                num_hidden_layers=32,
                projection_dim=1280
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
        model.to(torch.device("cpu"), dtype)

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
                    strength = self.parent.get_strength(i, hn.net_name)
                    if strength:
                        x[i] += ((v[i] * strength)).clone()
            out = self.original_forward(x)
            for lora in self.loras:
                v = None
                for i in range(len(self.parent.strength)):
                    strength = self.parent.get_strength(i, lora.net_name)
                    if strength:
                        if v == None:
                            v = lora(x)
                        out[i] += v[i] * strength
            return out

        def attach_lora(self, module, static):
            if static:
                weight = module.get_weight()
                strength =  self.parent.get_strength(0, module.net_name)
                self.original_module.weight += weight * strength
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
        self.strength_override = {}
        self.static = {}

        model_type = str(type(model))
        if "CLIP" in model_type:
            self.modules = self.hijack_model(model, 'te', ["CLIPAttention", "CLIPMLP"])
        elif "UNET" in model_type:
            self.modules = self.hijack_model(model, 'unet', ["Transformer2DModel", "Attention"])
        else:
            raise ValueError(f"INVALID TARGET {model_type}")
        self.model_type = model_type

    def clear(self):
        for name in self.modules:
            self.modules[name].clear()
        self.strength = {}
    
    def get_strength(self, index, name):
        if name in self.strength_override:
            return self.strength_override[name]
        if name in self.strength[index]:
            return self.strength[index][name]
        return 0.0

    def set_strength(self, strength):
        self.strength = strength

    def set_strength_override(self, name, strength):
        self.strength_override[name] = strength

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
            cn = ControlNet("CN-v1", dtype)
            missing, _ = cn.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(f"missing keys in ControlNet ({name}): " + ", ".join(missing))
        return cn

    @staticmethod
    def get_config(model_type):
        if model_type == "CN-v1":
            config = dict(
                cross_attention_dim=768
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config