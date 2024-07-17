import os
import torch
import utils

from lora import LycorisNetwork
from detailer import ADetailer

from transformers import CLIPTextConfig, CLIPTokenizer
from clip import CustomCLIP, CustomSDXLCLIP
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.controlnet import ControlNetModel

import accelerate
import accelerate.utils.modeling
def load_state_dict_in_place(model, state_dict):
    model_keys = [k for k, _ in model.named_parameters()]

    for key in model_keys:
        if not key in state_dict:
            continue
        accelerate.utils.modeling.set_module_tensor_to_device(model, key, state_dict[key].device, value=state_dict[key])

    missing_keys = [k for k in model_keys if not k in state_dict]
    unexpected_keys = [k for k in state_dict if not k in model_keys]
    return missing_keys, unexpected_keys

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
        if self.additional:
            if not 'added_cond_kwargs' in kwargs:
                kwargs['added_cond_kwargs'] = {}
            kwargs['cross_attention_kwargs'] = {'upcast_attention': self.upcast_attention}

            if 'added_cross_kwargs' in kwargs:
                for k,v in kwargs['added_cross_kwargs'].items():
                    kwargs['cross_attention_kwargs'][k] = v
                del kwargs['added_cross_kwargs']

        return super().__call__(*args, **kwargs)
        
    @staticmethod
    def from_model(name, state_dict, dtype=None, device="cpu"):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']
        prediction_type = state_dict['metadata']['prediction_type']
        model_variant = state_dict['metadata'].get('model_variant', "")
        del state_dict['metadata']

        utils.cast_state_dict(state_dict, dtype, device)

        with accelerate.init_empty_weights():
            unet = UNET(model_type, model_variant, prediction_type, dtype)

        missing, _ = load_state_dict_in_place(unet, state_dict)
        if missing:
            raise ValueError("Missing keys in UNET: " + ", ".join(missing))
        
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
                attention_head_dim=8
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
            #print('UPCASTING ATTENTION')
            self.upcast_attention = True
            self.model_variant = "SDv2.1"
            test_pred = self(test_latent, test_timestep, encoder_hidden_states=test_cond).sample

        is_v = (test_pred - 0.5).mean().item() < -1
        self.prediction_type = "v" if is_v else "epsilon"
        #print("DETECTED", self.prediction_type, "PREDICTION")

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
    def from_model(name, state_dict, dtype=None, device="cpu"):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']
        del state_dict['metadata']

        utils.cast_state_dict(state_dict, dtype, device)

        with accelerate.init_empty_weights():
            vae = VAE(model_type, dtype)

        missing, _ = load_state_dict_in_place(vae, state_dict)
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
        self.textual_inversions = []

    def encode(self, input_ids, clip_skip):
        return self.model(input_ids, clip_skip)
    
    def state_dict(self):
        return self.model.state_dict()

    def get_lora_model(self):
        if self.model_type == "SDXL-Base":
            return [self.model.ldm_clip, self.model.open_clip]
        else:
            return self.model

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)

    @staticmethod
    def from_model(name, state_dict, dtype=None, device="cpu"):
        if not dtype:
            dtype = state_dict['metadata']['dtype']
        model_type = state_dict['metadata']['model_type']
        del state_dict["metadata"]

        utils.cast_state_dict(state_dict, dtype, device)

        with accelerate.init_empty_weights():
            clip = CLIP(model_type, dtype)

        missing, _ = load_state_dict_in_place(clip.model, state_dict)
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

class LoRA(LycorisNetwork):
    @staticmethod
    def from_model(name, state_dict, dtype=None):
        if not dtype:
            dtype = state_dict['metadata']['dtype']

        utils.cast_state_dict(state_dict, dtype)

        net_name = "lora:" + name.rsplit(".",1)[0].rsplit(os.path.sep,1)[-1]
        model = LoRA(net_name)
        model.from_state_dict(state_dict)
        model.to(torch.device("cpu"), dtype)

        return model
    
    def __getattr__(self, name):
        if name == "device":
            return self.get_device()
        return super().__getattr__(name)

class AdditionalNetworks():
    def __init__(self, model):
        self.attached_dynamic = {}
        self.attached_static = {}

        self.strength = {}
        self.strength_override = {}

        self.model_type = "UNET" if type(model) == UNET else "CLIP"

    def clear(self):
        self.strength = {}
        self.strength_override = {}

        for _, net in self.attached_dynamic.items():
            net.detach_unet()
            net.detach_text_encoder()
        self.attached_dynamic = {}

    def has(self, net):
        if net.net_name in self.attached_dynamic:
            return True
        if net.net_name in self.attached_static:
            if self.get_strength(net.net_name) == net.strength:
                return True
        return False

    def attach(self, net, is_static):
        strength = self.get_strength(net.net_name)
        net.set_strength(strength)
        if is_static:
            if net.net_name in self.attached_static:
                return
            self.attached_static[net.net_name] = net
            if self.model_type == "UNET":
                net.merge_unet()
            elif self.model_type == "CLIP":
                net.merge_text_encoder()
        else:
            if net.net_name in self.attached_dynamic:
                return
            self.attached_dynamic[net.net_name] = net
            if self.model_type == "UNET":
                net.attach_unet()
            elif self.model_type == "CLIP":
                net.attach_text_encoder()

    def get_strength(self, name):
        strength = self.strength.get(name, 0.0)
        strength = self.strength_override.get(name, strength)
        return strength

    def set_strength(self, strength):
        self.strength = strength[0]
        for name, net in self.attached_dynamic.items():
            net.set_strength(self.get_strength(name))

    def set_strength_override(self, name, strength):
        self.strength_override[name] = strength

    def need_reset(self, allowed):
        attached = {name: self.get_strength(name) for name in self.attached_static}
        for name in attached:
            if name in allowed:
                if allowed[name] != attached[name]:
                    return True
            else:
                return True
        return False
    
    def reset(self):
        for name, net in self.attached_static.items():
            net.reset()
        self.attached_static = {}

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
            typ = "CN-v1"
            if name == "Anyline":
                typ = "CN-XL"
            cn = ControlNet(typ, dtype)
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
        elif model_type == "CN-XL":
            config = dict(
                cross_attention_dim=2048,
                addition_embed_type="text_time",
                addition_embed_type_num_heads=64,
                addition_time_embed_dim=256,
                attention_head_dim=[5,10,20],
                block_out_channels=[320,640,1280],
                down_block_types=[
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D"
                ],
                mid_block_type="UNetMidBlock2DCrossAttn",
                projection_class_embeddings_input_dim=2816,
                transformer_layers_per_block=[1,2,10],
                use_linear_projection=True
            )
        else:
            raise ValueError(f"unknown type: {model_type}")
        return config
    
class Detailer(ADetailer):
    def from_model(name, model, dtype=None):
        model.name = name
        return model