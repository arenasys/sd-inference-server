import torch
import torchvision.transforms as transforms
import PIL
import random
import io
import os
import safetensors.torch
import time
import shutil
import tomesd
import contextlib

DIRECTML_AVAILABLE = False
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except:
    pass

import prompts
import samplers_k
import samplers_ddpm
import guidance
import utils
import storage
import upscalers
import inference
import convert
import attention
import controlnet
import preview

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler a", "clip_skip": 1, "eta": 1,
    "hr_upscaler": "Latent (nearest)", "hr_strength": 0.7, "img2img_upscaler": "Lanczos", "mask_blur": 4
}

TYPES = {
    int: ["width", "height", "steps", "seed", "batch_size", "clip_skip", "mask_blur", "hr_steps", "padding"],
    float: ["scale", "eta", "hr_factor", "hr_eta"],
}

SAMPLER_CLASSES = {
    "Euler": samplers_k.Euler,
    "Euler a": samplers_k.Euler_a,
    "DPM++ 2M": samplers_k.DPM_2M,
    "DPM++ 2S a": samplers_k.DPM_2S_a,
    "DPM++ SDE": samplers_k.DPM_SDE,
    "DPM++ 2M Karras": samplers_k.DPM_2M_Karras,
    "DPM++ 2S a Karras": samplers_k.DPM_2S_a_Karras,
    "DPM++ SDE Karras": samplers_k.DPM_SDE_Karras,
    "DDIM": samplers_ddpm.DDIM,
    "PLMS": samplers_ddpm.PLMS
}

UPSCALERS_LATENT = {
    "Latent (bicubic)": transforms.InterpolationMode.BICUBIC,
    "Latent (bilinear)": transforms.InterpolationMode.BILINEAR,
    "Latent (nearest)": transforms.InterpolationMode.NEAREST,
}

UPSCALERS_PIXEL = {
    "Lanczos": transforms.InterpolationMode.LANCZOS,
    "Bicubic": transforms.InterpolationMode.BICUBIC,
    "Bilinear": transforms.InterpolationMode.BILINEAR,
    "Nearest": transforms.InterpolationMode.NEAREST,
}

CROSS_ATTENTION = {
    "Default": attention.use_optimized_attention,
    "Split v1": attention.use_split_attention_v1,
    "Split v2": attention.use_split_attention,
    "Flash": attention.use_flash_attention,
    "Diffusers": attention.use_diffusers_attention,
    "xFormers": attention.use_xformers_attention,
}

def format_float(x):
    return f"{x:.4f}".rstrip('0').rstrip('.')

def model_name(x):
    return x.rsplit('.',1)[0].rsplit(os.path.sep,1)[-1]

class GenerationParameters():
    def __init__(self, storage: storage.ModelStorage, device):
        self.storage = storage
        self.device = device

        self.device_names = []
        for i in range(torch.cuda.device_count()):
            original = torch.cuda.get_device_name(i)
            name = original
            i = 2
            while name in self.device_names:
                name = original + f" ({i})"
            self.device_names += [name]
        
        if DIRECTML_AVAILABLE:
            self.device_names += ["DirectML"]

        self.device_names += ["CPU"]

        self.last_models_modified = False
        self.last_models_config = None

        self.callback = None

    def set_status(self, status):
        if self.callback:
            if not self.callback({"type": "status", "data": {"message": status}}):
                raise RuntimeError("Aborted")

    def set_progress(self, progress):
        if self.callback:
            if not self.callback({"type": "progress", "data": progress}):
                raise RuntimeError("Aborted")

    def on_step(self, progress, latents):
        step = self.current_step
        total = self.total_steps
        rate = progress["rate"]
        remaining = (total - step) / rate if rate and total else 0

        if "n" in progress:
            self.current_step += 1
        
        progress = {"current": step, "total": total, "rate": rate, "remaining": remaining}
        
        interval = int(self.preview_interval or 0)
        if latents != None and self.show_preview and step % interval == 0:
            if self.show_preview == "Full":
                images = preview.full_preview(latents / 0.18215, self.vae)
            elif self.show_preview == "Medium":
                images = preview.model_preview(latents)
            else:
                images = preview.cheap_preview(latents)
            for i in range(len(images)):
                bytesio = io.BytesIO()
                images[i].save(bytesio, format='PNG')
                images[i] = bytesio.getvalue()
            progress["previews"] = images

        self.set_progress(progress)

    def on_artifact(self, name, images):
        if self.callback:
            if type(images[0]) == list:
                for j in range(max([len(i) for i in images])):
                    images_data = []
                    for i in range(len(images)):
                        if j < len(images[i]):
                            bytesio = io.BytesIO()
                            images[i][j].save(bytesio, format='PNG')
                            images_data += [bytesio.getvalue()]
                        else:
                            images_data += [None]
                    self.callback({"type": "artifact", "data": {"name": f"{name} {j+1}", "images": images_data}})
            else:
                images_data = []
                for i in images:
                    if i != None:
                        bytesio = io.BytesIO()
                        i.save(bytesio, format='PNG')
                        images_data += [bytesio.getvalue()]
                    else:
                        images_data += [None]
                self.callback({"type": "artifact", "data": {"name": name, "images": images_data}})
        
    def on_complete(self, images, metadata):
        if self.callback:
            self.set_status("Fetching")
            images_data = []
            for i in images:
                bytesio = io.BytesIO()
                i.save(bytesio, format='PNG')
                images_data += [bytesio.getvalue()]
            self.callback({"type": "result", "data": {"images": images_data, "metadata": metadata}})
            
    def reset(self):
        for attr in list(self.__dict__.keys()):
            if not attr in ["storage", "device", "device_names", "callback", "last_models_modified", "last_models_config"]:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key == "image" or key == "cn_image":
                for i in range(len(value)):
                    if type(value[i]) == bytes or type(value[i]) == bytearray:
                        value[i] = PIL.Image.open(io.BytesIO(value[i]))
                        if value[i].mode == 'RGBA':
                            value[i] = PIL.Image.alpha_composite(PIL.Image.new('RGBA',value[i].size,(0,0,0)), value[i])
                            value[i] = value[i].convert("RGB")
                        if value[i].mode != 'RGB':
                            value[i] = value[i].convert("RGB")
            if key == "mask":
                for i in range(len(value)):
                    if type(value[i]) == bytes or type(value[i]) == bytearray:
                        value[i] = PIL.Image.open(io.BytesIO(value[i]))
                        if value[i].mode == 'RGBA':
                            value[i] = value[i].split()[-1]
                        else:
                            value[i] = value[i].convert("L")
            if key == "area":
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        if type(value[i][j]) == bytes or type(value) == bytearray:
                            value[i][j] = PIL.Image.open(io.BytesIO(value[i][j]))
                            if value[i][j].mode == 'RGBA':
                                value[i][j] = value[i][j].split()[-1]
                            else:
                                value[i][j] = value[i][j].convert("L")
            
            setattr(self, key, value)
    
    def check_modified(self, nets):
        current = tuple([self.network_mode, self.unet, self.clip, *nets])
        self.reattach_networks = current != self.last_models_config
        if self.reattach_networks and self.last_models_modified:
            self.storage.clear_modified()
            self.last_models_modified = False
        if self.network_mode == "Dynamic":
            self.reattach_networks = True
        self.last_models_config = current

    def load_models(self):
        if not self.unet or type(self.unet) == str:
            self.unet_name = self.unet or self.model
            self.set_status("Loading UNET")
            self.unet = self.storage.get_unet(self.unet_name, self.device)
        
        if not self.clip or type(self.clip) == str:
            self.clip_name = self.clip or self.model
            self.set_status("Loading CLIP")
            self.clip = self.storage.get_clip(self.clip_name, self.device)
            self.clip.set_textual_inversions(self.storage.get_embeddings(self.device))
        
        if not self.vae or type(self.vae) == str:
            self.vae_name = self.vae or self.model
            self.set_status("Loading VAE")
            self.vae = self.storage.get_vae(self.vae_name, self.device)
        
        self.storage.clear_file_cache()
        
        self.storage.enforce_network_limit(self.cn or [], "CN")
        if self.cn:
            self.set_status("Loading ControlNet")
            self.cn = [self.storage.get_controlnet(cn, self.device) for cn in self.cn]

        if self.hr_upscaler:
            if not self.hr_upscaler in UPSCALERS_LATENT and not self.hr_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.hr_upscaler, self.device)

        if self.img2img_upscaler:
            if not self.img2img_upscaler in UPSCALERS_LATENT and not self.img2img_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.img2img_upscaler, self.device)

    def need_models(self, unet, vae, clip):
        if not self.minimal_vram:
            return

        if unet:
            self.storage.load(self.unet, self.device)
        else:
            self.storage.unload(self.unet)
        
        if vae:
            self.storage.load(self.vae, self.device)
        else:
            self.storage.unload(self.vae)
        
        if clip:
            self.storage.load(self.clip, self.device)
        else:
            self.storage.unload(self.clip)

    def check_parameters(self, required, optional):
        missing, unused = list(required), []
        other = "storage, device, model".split(", ")

        for attr, value in DEFAULTS.items():
            if getattr(self, attr) == None:
                setattr(self, attr, value)

        for attr in self.__dict__.keys():
            if attr in required:
                missing.remove(attr)
                continue
            if attr in optional or attr in other:
                continue
            unused += [attr]

        for t in TYPES:
            for attr in TYPES[t]:
                if getattr(self, attr) != None:
                    setattr(self, attr, t(getattr(self, attr)))

        if missing:
            raise ValueError(f"missing required parameters: {', '.join(missing)}")

        if not self.sampler in SAMPLER_CLASSES:
            raise ValueError(f"unknown sampler: {self.sampler}")

        if (self.width or self.height) and not (self.width and self.height):
            raise ValueError("width and height must both be set")
        
        if self.minimal_vram and self.show_preview == "Full":
            raise ValueError("Full preview is incompatible with minimal VRAM")
        
        if self.attention and self.attention in CROSS_ATTENTION:
            CROSS_ATTENTION[self.attention](self.device)

    def set_device(self):
        device = torch.device("cuda")
        if self.device_name in self.device_names:
            idx = self.device_names.index(self.device_name)
            if self.device_name == "CPU":
                device = torch.device("cpu")
                self.storage.dtype = torch.float32
            elif self.device_name == "DirectML":
                device = torch_directml.device()
                self.storage.dtype = torch.float16
            else:
                device = torch.device(idx)
                self.storage.dtype = torch.float16
        self.device = device

    def listify(self, *args):
        if args == None:
            return (None,)
        args = [[a] if type(a) != list else a for a in args]
        return args

    @torch.inference_mode()
    def upscale_latents(self, latents, mode, width, height, seeds):
        if mode in UPSCALERS_LATENT:
            return upscalers.upscale_single(latents, UPSCALERS_LATENT[mode], width//8, height//8)

        images = utils.decode_images(self.vae, latents)
        if mode in UPSCALERS_PIXEL:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[mode], width, height) 
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)
        
        self.set_status("Encoding")
        return utils.encode_images(self.vae, seeds, images)

    def upscale_images(self, vae, images, mode, width, height, seeds):
        if mode in UPSCALERS_LATENT:
            latents = utils.get_latents(vae, seeds, images)
            latents = upscalers.upscale_single(latents, UPSCALERS_LATENT[mode], width//8, height//8)

            if type(latents) == list:
                latents = torch.stack(latents)

            return latents, None
        
        if mode in UPSCALERS_PIXEL:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[mode], width, height)
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)

        self.set_status("Encoding")
        latents = utils.get_latents(vae, seeds, images)
        return latents, images

    def get_seeds(self, batch_size):
        (seeds,) = self.listify(self.seed)
        if self.subseed:
            subseeds = []
            if type(self.subseed[0]) in {tuple, list}:
                subseeds = self.subseed
            else:
                subseeds = [self.subseed]
            subseeds = [tuple(s) for s in subseeds]

            for i in range(len(subseeds)):
                a, b = subseeds[i]
                subseeds[i] = (int(a), float(b))
        else:
            subseeds = [(0,0)]

        for i in range(len(seeds)):
            if seeds[i] == -1:
                seeds[i] = random.randrange(2147483646)
        for i in range(len(subseeds)):
            if subseeds[i][0] == -1:
                subseeds[i] = (random.randrange(2147483646), subseeds[i][1])

        if len(seeds) < batch_size:
            last_seed = seeds[-1]
            seeds += [last_seed + i + 1 for i in range(batch_size-len(seeds))]

        if len(subseeds) < batch_size:
            last_seed, last_strength = subseeds[-1]
            subseeds += [(last_seed + i + 1, last_strength) for i in range(batch_size-len(subseeds))]

        return seeds, subseeds
    
    def get_batch_size(self):
        batch_size = max(self.batch_size or 1, 1)
        for i in [self.prompt, self.negative_prompt, self.seeds, self.subseeds, self.image, self.mask, self.area]:
            if i != None and type(i) == list:
                batch_size = max(batch_size, len(i))
        return batch_size
    
    def get_metadata(self, mode, width, height, batch_size, prompts=None, seeds=None, subseeds=None):
        metadata = []

        for i in range(batch_size):
            m = {
                "mode": mode,
                "width": width,
                "height": height
            }

            if subseeds != None:
                sds = []
                strs = []
                active = False
                for s, r in subseeds:
                    if r != 0.0:
                        active = True
                    sds += [str(s)]
                    strs += [format_float(r)]
                if active:
                    m["subseed"] = ", ".join(sds)
                    m["subseed_strength"] = ", ".join(strs)

            if self.eta != DEFAULTS["eta"]:
                m["eta"] = format_float(self.eta)

            if mode in {"txt2img", "img2img"}:
                if len({self.unet_name, self.clip_name, self.vae_name}) == 1:
                    m["model"] = model_name(self.unet_name)
                else:
                    m["UNET"] = model_name(self.unet_name)
                    m["CLIP"] = model_name(self.clip_name)
                    m["VAE"] = model_name(self.vae_name)
                m["prompt"] = ' AND '.join(prompts[i][0])
                m["negative_prompt"] = ' AND '.join(prompts[i][1])
                m["seed"] = seeds[i]
                m["steps"] = self.steps
                m["scale"] = format_float(self.scale)
                m["sampler"] = self.sampler
                m["clip_skip"] = self.clip_skip

            if mode == "img2img":
                m["strength"] = format_float(self.strength)
                m["img2img_upscaler"] = model_name(self.img2img_upscaler)
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            if mode == "txt2img":
                if self.hr_factor and self.hr_factor != 1.0:
                    m["hr_factor"] = format_float(self.hr_factor)
                    m["hr_upscaler"] =  model_name(self.hr_upscaler)
                    m["hr_strength"] = format_float(self.hr_strength)
                    if self.hr_steps and self.hr_steps != self.steps:
                        m["hr_steps"] = self.hr_steps
                    if self.hr_sampler and self.hr_sampler != self.sampler:
                        m["hr_sampler"] = self.hr_sampler
                    if self.hr_eta and self.hr_eta != self.eta:
                        m["hr_eta"] = format_float(self.hr_eta)
            
            if mode == "upscale":
                m["img2img_upscaler"] = model_name(self.img2img_upscaler)
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            metadata += [m]

        return metadata
    
    def attach_networks(self, all_nets, unet_nets, clip_nets, device):
        self.detach_networks()

        static = self.network_mode == "Static"

        if static:
            self.unet.additional.set_strength([unet_nets])
            self.clip.additional.set_strength([clip_nets])

        lora_names = []
        hn_names = []

        for n in all_nets:
            prefix, n = n.split(":",1)
            if prefix == "lora":
                lora_names += [n]
            if prefix == "hypernet":
                hn_names += [n]

        if lora_names:
            if self.reattach_networks:
                self.set_status("Loading LoRAs")
                self.storage.enforce_network_limit(lora_names, "LoRA")
                self.loras = [self.storage.get_lora(name, device) for name in lora_names]

                for lora in self.loras:
                    lora.attach([self.unet.additional, self.clip.additional], static)
                    if static:
                        self.last_models_modified = True
                        lora.to("cpu")
        else:
            self.storage.enforce_network_limit([], "LoRA")

        if hn_names:
            self.set_status("Loading Hypernetworks")
            self.storage.enforce_network_limit(hn_names, "HN")
            self.hns = [self.storage.get_hypernetwork(name, device) for name in hn_names]

            for hn in self.hns:
                hn.attach(self.unet.additional)
        else:
            self.storage.enforce_network_limit([], "HN")

    def detach_networks(self):
        self.unet.additional.clear()
        self.clip.additional.clear()

    def attach_tome(self, HR=False):
        unet = self.unet
        if type(unet) == controlnet.ControlledUNET:
            unet = unet.unet

        ratio = self.hr_tome_ratio if HR else self.tome_ratio

        if ratio:
            tomesd.apply_patch(unet, ratio=ratio)
        else:
            tomesd.remove_patch(unet)

    def get_autocast_context(self, autocast, device):
        if autocast:
            return torch.autocast('cpu' if device == torch.device('cpu') else 'cuda')
        return contextlib.nullcontext()

    def prepare_images(self, inputs, extents, width, height):
        if not inputs:
            return inputs
        outputs = utils.apply_extents(inputs, extents)
        outputs = upscalers.upscale(outputs, transforms.InterpolationMode.NEAREST, width, height)
        return outputs

    @torch.inference_mode()
    def txt2img(self):
        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, prompt, width, height, seed, scale, steps".split(", ")
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscaler, hr_strength, hr_sampler, hr_eta, area".split(", ")
        self.check_parameters(required, optional)
    
        self.set_status("Parsing")
        conditioning = prompts.BatchedConditioningSchedules(self.prompt, self.steps, self.clip_skip)
        self.check_modified(conditioning.get_initial_networks(True))

        self.set_status("Loading")
        self.set_device()
        self.load_models()

        self.attach_tome()

        self.set_status("Configuring")
        batch_size = self.get_batch_size()
        device = self.unet.device

        if self.cn:
            self.unet = controlnet.ControlledUNET(self.unet, self.cn)
            dtype = self.unet.dtype

            annotators = [self.storage.get_controlnet_annotator(o["annotator"], device, dtype) for o in self.cn_opts]
            scales = [o["scale"] for o in self.cn_opts]
            args = [o["args"] for o in self.cn_opts]

            cn_cond, cn_outputs = controlnet.preprocess_control(self.cn_image, annotators, args, scales)
            if self.keep_artifacts:
                self.on_artifact("Control", [cn_outputs]*batch_size)
            self.unet.set_controlnet_conditioning(cn_cond)

        seeds, subseeds = self.get_seeds(batch_size)
        
        self.current_step = 0
        self.total_steps = self.steps

        if self.hr_factor:
            self.hr_steps = self.hr_steps or self.steps
            self.hr_sampler = self.hr_sampler or self.sampler
            self.hr_eta = self.hr_eta or self.eta
            self.total_steps += self.hr_steps

        metadata = self.get_metadata("txt2img", self.width, self.height, batch_size, self.prompt, seeds, subseeds)

        if self.area:
            if self.keep_artifacts:
                self.on_artifact("Area", self.area)
            area = utils.preprocess_areas(self.area, self.width, self.height)
        else:
            area = []
        
        self.set_status("Attaching")
        self.attach_networks(*conditioning.get_initial_networks(), device)

        self.set_status("Encoding")
        conditioning.encode(self.clip, area)

        self.set_status("Preparing")
        self.need_models(unet=True, vae=False, clip=False)
        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, self.width // 8, self.height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.unet.inpainting: #SDv1-Inpainting, SDv2-Inpainting
            images = torch.zeros((batch_size, 3, self.width, self.height))
            inpainting_masked, inpainting_masks = utils.encode_inpainting(images, None, self.vae, seeds)
            denoiser.set_inpainting(inpainting_masked, inpainting_masks)
        
        self.set_status("Generating")

        with self.get_autocast_context(self.autocast, device):
            latents = inference.txt2img(denoiser, sampler, noise, self.steps, self.on_step)

        self.need_models(unet=False, vae=True, clip=False)

        if not self.hr_factor:
            self.set_status("Decoding")
            images = utils.decode_images(self.vae, latents)
            self.on_complete(images, metadata)
            self.need_models(unet=False, vae=False, clip=False)
            return images

        self.set_status("Preparing")
        if self.keep_artifacts:
            images = utils.decode_images(self.vae, latents)
            self.on_artifact("Base", images)

        width = int(self.width * self.hr_factor)
        height = int(self.height * self.hr_factor)

        self.need_models(unet=False, vae=True, clip=True)

        denoiser.reset()
        sampler = SAMPLER_CLASSES[self.hr_sampler](denoiser, self.hr_eta)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)

        conditioning.switch_to_HR(self.hr_steps)

        if self.network_mode == "Dynamic":
            self.set_status("Attaching")
            self.attach_networks(*conditioning.get_initial_networks(), device)

        if area:
            area = utils.preprocess_areas(self.area, width, height)

        self.set_status("Encoding")
        conditioning.encode(self.clip, area)

        self.need_models(unet=False, vae=True, clip=False)

        self.set_status("Upscaling")
        latents = self.upscale_latents(latents, self.hr_upscaler, width, height, seeds)

        if self.keep_artifacts:
            images = utils.decode_images(self.vae, latents)
            self.on_artifact("Upscaled", images)

        self.need_models(unet=True, vae=False, clip=False)

        self.attach_tome(HR=True)

        if self.cn:
            self.cn_image = upscalers.upscale(self.cn_image, transforms.InterpolationMode.NEAREST, width, height)
            cn_cond, cn_outputs = controlnet.preprocess_control(self.cn_image, annotators, args, scales)
            self.unet.set_controlnet_conditioning(cn_cond)
            if self.keep_artifacts:
                self.on_artifact("Control HR", [cn_outputs]*batch_size)

        self.set_status("Generating")

        with self.get_autocast_context(self.autocast, device):
            latents = inference.img2img(latents, denoiser, sampler, noise, self.hr_steps, True, self.hr_strength, self.on_step)

        self.set_status("Decoding")
        self.need_models(unet=False, vae=True, clip=False)
        images = utils.decode_images(self.vae, latents)

        self.on_complete(images, metadata)

        self.need_models(unet=False, vae=False, clip=False)
        return images

    @torch.inference_mode()
    def img2img(self):
        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, image, prompt, seed, scale, steps, strength".split(", ")
        optional = "img2img_upscaler, mask, mask_blur, clip_skip, eta, batch_size, padding, width, height".split(", ")
        self.check_parameters(required, optional)

        self.set_status("Parsing")
        conditioning = prompts.BatchedConditioningSchedules(self.prompt, self.steps, self.clip_skip)
        self.check_modified(conditioning.get_initial_networks(True))

        self.set_status("Loading")
        self.set_device()
        self.load_models()

        self.attach_tome()
        
        self.set_status("Preparing")
        batch_size = self.get_batch_size()
        device = self.unet.device
        images = self.image
        masks = (self.mask or []).copy()
        width, height = self.width, self.height

        extents = utils.get_extents(images, masks, self.padding, width, height)
        for i in range(len(masks)):
            if masks[i] == None:
                masks[i] = PIL.Image.new("L", (images[i].width, images[i].height), color=255)

        original_images = images
        images = utils.apply_extents(images, extents)
        masks = self.prepare_images(masks, extents, width, height)

        seeds, subseeds = self.get_seeds(batch_size)
        metadata = self.get_metadata("img2img",  width, height, batch_size, self.prompt, seeds, subseeds)

        if self.cn:
            self.unet = controlnet.ControlledUNET(self.unet, self.cn)
            dtype = self.unet.dtype

            annotators = [self.storage.get_controlnet_annotator(o["annotator"], device, dtype) for o in self.cn_opts]
            scales = [o["scale"] for o in self.cn_opts]
            args = [o["args"] for o in self.cn_opts]

            for i in range(len(self.cn_image)):
                if self.mask and self.mask[i] != None:
                    self.cn_image[i] = self.prepare_images([self.cn_image[i]], [extents[i]], width, height)[0]

            cn_cond, cn_outputs = controlnet.preprocess_control(self.cn_image, annotators, args, scales)
            if self.keep_artifacts:
                self.on_artifact("Control", [cn_outputs]*batch_size)
            self.unet.set_controlnet_conditioning(cn_cond)

        if self.area:
            for i in range(len(self.area)):
                if self.mask and self.mask[i] != None:
                    self.area[i] = self.prepare_images(self.area[i], [extents[i]]*len(self.area[i]), width, height)
            if self.keep_artifacts:
                self.on_artifact("Area", self.area)
            self.area = utils.preprocess_areas(self.area, width, height)
        else:
            self.area = []

        self.current_step = 0
        self.total_steps = int(self.steps * self.strength) + 1

        self.set_status("Attaching")
        self.attach_networks(*conditioning.get_initial_networks(), device)

        self.set_status("Encoding")
        conditioning.encode(self.clip, self.area)

        self.need_models(unet=True, vae=True, clip=False)

        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        self.set_status("Upscaling")
        latents, upscaled_images = self.upscale_images(self.vae, images, self.img2img_upscaler, width, height, seeds)
        original_latents = latents

        if self.keep_artifacts:
            if not upscaled_images:
                upscaled_images = utils.decode_images(self.vae, latents)
            self.on_artifact("Input", upscaled_images)

        if self.mask:
            self.set_status("Preparing")
            masks = [None if mask == None else mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]
            mask_latents = utils.get_masks(device, masks)

            if self.mask_fill == "Noise":
                latents = latents * ((mask_latents * self.strength) + (1-self.strength))
            
            denoiser.set_mask(mask_latents, original_latents)
            if self.keep_artifacts:
                self.on_artifact("Mask", [masks[i] if self.mask[i] else None for i in range(len(masks))])

        if self.unet.inpainting: #SDv1-Inpainting, SDv2-Inpainting
            self.set_status("Preparing")
            if not self.mask:
                masks = None
            inpainting_masked, inpainting_masks = utils.encode_inpainting(upscaled_images, masks, self.vae, seeds)
            denoiser.set_inpainting(inpainting_masked, inpainting_masks)

        self.need_models(unet=True, vae=False, clip=False)

        self.set_status("Generating")

        with self.get_autocast_context(self.autocast, device):
            latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        self.need_models(unet=False, vae=True, clip=False)

        self.set_status("Decoding")
        images = utils.decode_images(self.vae, latents)

        self.need_models(unet=False, vae=False, clip=False)

        if self.mask:
            outputs, masked = utils.apply_inpainting(images, original_images, masks, extents)
            if self.keep_artifacts:
                self.on_artifact("Output", [masked[i] if self.mask[i] else None for i in range(len(masked))])
            images = [outputs[i] if self.mask[i] else images[i] for i in range(len(images))]

        self.on_complete(images, metadata)
        return images
    
    @torch.inference_mode()
    def upscale(self):
        self.set_status("Loading")
        self.set_device()
        self.storage.clear_file_cache()
        if not self.img2img_upscaler in UPSCALERS_PIXEL and not self.img2img_upscaler in UPSCALERS_LATENT:
            self.set_status("Loading Upscaler")
            self.upscale_model = self.storage.get_upscaler(self.img2img_upscaler, self.device)

        self.set_status("Configuring")
        required = "image, img2img_upscaler, width, height".split(", ")
        optional = "mask, mask_blur, padding".split(", ")
        self.check_parameters(required, optional)

        batch_size = self.get_batch_size()

        images = self.image
        original_images = images

        if self.mask:
            masks = self.mask

        width, height = self.width, self.height

        metadata = self.get_metadata("upscale", width, height, batch_size)

        if self.mask:
            self.set_status("Preparing")
            images, masks, extents = utils.prepare_inpainting(images, masks, self.padding, width, height)
            masks = [mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]

        self.set_status("Upscaling")
        if not self.upscale_model:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[self.img2img_upscaler], width, height)
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)

        if self.mask:
            images = utils.apply_inpainting(images, original_images, masks, extents)

        self.on_complete(images, metadata)
        return images
    
    @torch.inference_mode()
    def annotate(self):
        self.set_status("Loading")
        self.set_device()
        self.storage.clear_file_cache()

        device = self.device
        dtype = torch.float16

        self.set_status("Annotating")
        annotator = self.storage.get_controlnet_annotator(self.cn_annotator[0], device, dtype)
        _, img = controlnet.annotate(self.cn_image[0], annotator, self.cn_args[0])

        self.set_status("Fetching")
        bytesio = io.BytesIO()
        img.save(bytesio, format='PNG')
        data = bytesio.getvalue()
        if self.callback:
            if not self.callback({"type": "annotate", "data": {"images": [data]}}):
                raise RuntimeError("Aborted")
    
    def options(self):
        self.storage.find_all()
        data = {"sampler": list(SAMPLER_CLASSES.keys())}
        for k in self.storage.files:
            data[k] = list(self.storage.files[k].keys())

        data["hr_upscaler"] = list(UPSCALERS_LATENT.keys()) + list(UPSCALERS_PIXEL.keys()) + data["SR"]
        data["img2img_upscaler"] = list(UPSCALERS_PIXEL.keys()) + list(UPSCALERS_LATENT.keys()) + data["SR"]

        available = attention.get_available()
        data["attention"] = [k for k,v in CROSS_ATTENTION.items() if v in available]
        data["TI"] = list(self.storage.embeddings_files.keys())
        data["device"] = self.device_names
        
        if self.callback:
            if not self.callback({"type": "options", "data": data}):
                raise RuntimeError("Aborted")
    
    @torch.inference_mode()
    def manage(self):
        self.set_status("Configuring")
        self.check_parameters(["operation"], [])

        if self.operation == "build":
            self.build(self.file)
        elif self.operation == "modify":
            self.modify(self.old_file, self.new_file)
        elif self.operation == "prune":
            self.prune(self.file)
        elif self.operation == "rename":
            self.set_status("Renaming")
            os.makedirs(self.new_file.rsplit(os.path.sep,1)[0], exist_ok=True)
            shutil.move(self.old_file, self.new_file)
        elif self.operation == "move":
            self.set_status("Moving")
            for folder in storage.MODEL_FOLDERS[self.new_folder]:
                destination = os.path.join(self.storage.path, folder)
                if self.new_subfolder:
                    destination = os.path.join(destination, self.new_subfolder)
                if os.path.exists(destination):
                    break
            else:
                raise ValueError(f"failed to find destination")
            source = os.path.join(self.storage.path, self.old_file)
            destination = os.path.join(destination, self.old_file.rsplit(os.path.sep, 1)[-1])
            shutil.move(source, destination)
        else:
            raise ValueError(f"unknown operation: {self.operation}")
        
        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")
        
    def prune(self, file):
        old_file = file
        new_file = file

        on, oe = old_file.rsplit(".",1)
        if oe == "qst":
            return
        if not oe == "safetensors":
            new_file = on + ".safetensors"

        self.modify(old_file, new_file)

    def modify(self, old, new):
        o = os.path.join(self.storage.path, old)
        n = os.path.join(self.storage.path, new)

        if not new:
            self.set_status("Deleting")
            os.remove(o)
            return
        
        oe, ne = old.rsplit(".",1)[-1], new.rsplit(".",1)[-1]   
        ot, nt = old.split(os.path.sep, 1)[0], new.split(os.path.sep, 1)[0]

        component = ot in storage.MODEL_FOLDERS["SD"] and any([e in old for e in {".vae.",".clip.",".unet."}])
        if ot in storage.MODEL_FOLDERS["SD"] and not component:
            self.set_status("Converting")
            if not ne in {"qst", "safetensors"}:
                raise ValueError(f"unsuported checkpoint type: {ne}. supported types are: safetensors, qst")
            if oe != "qst":
                state_dict = convert.convert_checkpoint(o)
            else:
                state_dict = safetensors.torch.load_file(o)

            if ne != "qst":
                model_type = ''.join([chr(c) for c in state_dict["metadata.model_type"]])
                state_dict = convert.revert(model_type, state_dict)
            
            safetensors.torch.save_file(state_dict, n)
            if n != o:
                os.remove(o)
        else:
            self.set_status("Converting")
            if not ne in {"safetensors", "st"}:
                raise ValueError(f"unsuported model type: {ne}. supported types are: safetensors, qst")
            if oe in {"pt", "pth", "ckpt"}:
                state_dict = torch.load(o, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            elif oe in {"qst", "safetensors"}:
                state_dict = safetensors.torch.load_file(o)
            else:
                raise ValueError(f"unknown model type: {oe}")
            
            caution = False
            if component:
                caution = convert.clean_component(state_dict)
            else:
                for k in list(state_dict.keys()):
                    if ne in {"safetensors", "qst"} and type(state_dict[k]) != torch.Tensor:
                        del state_dict[k]
                        caution = True
                        continue   

            for k in list(state_dict.keys()):
                if state_dict[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                    state_dict[k] = state_dict[k].to(torch.float16)
            
            if len(state_dict) == 0:
                raise ValueError(f"conversion failed, empty model after pruning")
            
            if caution:
                trash_path = os.path.join(self.storage.path, "TRASH")
                trash = os.path.join(trash_path, o.rsplit(os.path.sep)[-1])
                os.makedirs(trash_path, exist_ok=True)
                if n == o:
                    shutil.copy(o, trash)
                else:
                    os.rename(o, trash)

            safetensors.torch.save_file(state_dict, n)

            if not caution and n != o:
                os.remove(o)

    def build(self, filename):
        file_type = filename.rsplit(".",1)[-1]
        if not file_type in {"safetensors"}:
            raise ValueError(f"unsuported checkpoint type: {file_type}. supported types are: safetensors")

        self.storage.clear_modified()
        self.network_mode = "Static"
        self.last_models_modified = False
        self.reattach_networks = True

        self.set_status("Parsing")
        if self.prompt:
            conditioning = prompts.BatchedConditioningSchedules(self.prompt, 1, 1)
        
        self.set_status("Loading")
        self.set_device()
        self.load_models()

        device = self.unet.device

        if self.prompt:
            self.set_status("Attaching")
            nets = conditioning.get_initial_networks()
            self.attach_networks(*nets, device)
        
        self.set_status("Building")

        state_dict = {}

        model_type = self.unet.model_type

        if self.clip.model_type != model_type:
            raise ValueError(f"UNET and CLIP are incompatible")

        comp_state_dict = self.unet.state_dict()
        comp_prefix = self.unet.model_type + ".UNET."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]

        comp_state_dict = self.clip.state_dict()
        comp_prefix = self.clip.model_type + ".CLIP."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]

        comp_state_dict = self.vae.state_dict()
        comp_prefix = self.vae.model_type + ".VAE."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]
        del comp_state_dict

        state_dict = convert.revert(model_type, state_dict)

        self.set_status(f"Saving")
        filename = os.path.join(self.storage.path, "SD", filename)

        safetensors.torch.save_file(state_dict, filename)