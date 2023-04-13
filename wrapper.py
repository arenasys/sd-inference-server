import torch
import torchvision.transforms as transforms
import PIL
import random
import io
import os
import safetensors.torch

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
    "xFormers": attention.use_xformers_attention
}

HAVE_XFORMERS = False
try:
    import xformers
    HAVE_XFORMERS = True
except Exception:
    pass

def format_float(x):
    return f"{x:.4f}".rstrip('0').rstrip('.')

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
        self.device_names += ["CPU"]

        self.callback = None

    def set_status(self, status):
        if self.callback:
            if not self.callback({"type": "status", "data": {"message": status}}):
                raise RuntimeError("Aborted")

    def set_progress(self, progress):
        if self.callback:
            if not self.callback({"type": "progress", "data": progress}):
                raise RuntimeError("Aborted")

    def on_step(self, progress):
        step = self.current_step
        total = self.total_steps
        rate = progress["rate"]
        remaining = (total - step) / rate if rate and total else 0

        if "n" in progress:
            self.current_step += 1

        progress = {"current": step, "total": total, "rate": rate, "remaining": remaining}

        self.set_progress(progress)
    
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
            if not attr in ["storage", "device", "device_names", "callback"]:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key == "image" or key == "mask":
                if type(value) == bytes or type(value) == bytearray:
                    value = PIL.Image.open(io.BytesIO(value))
                    if value[i].mode == 'RGBA':
                        value = PIL.Image.alpha_composite(PIL.Image.new('RGBA',value.size,(0,0,0)), value)
                        value = value.convert("RGB")
                if type(value) == list:
                    for i in range(len(value)):
                        if type(value[i]) == bytes or type(value[i]) == bytearray:
                            value[i] = PIL.Image.open(io.BytesIO(value[i]))
                            if value[i].mode == 'RGBA':
                                value[i] = PIL.Image.alpha_composite(PIL.Image.new('RGBA',value[i].size,(0,0,0)), value[i])
                                value[i] = value[i].convert("RGB")

            setattr(self, key, value)
    
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

        if self.hr_upscaler:
            if not self.hr_upscaler in UPSCALERS_LATENT and not self.hr_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.hr_upscaler, self.device)

        if self.img2img_upscaler:
            if not self.img2img_upscaler in UPSCALERS_LATENT and not self.img2img_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.img2img_upscaler, self.device)

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
        
        if self.attention and self.attention in CROSS_ATTENTION:
            CROSS_ATTENTION[self.attention](self.device)

    def set_device(self):
        device = torch.device("cuda")
        if self.device_name in self.device_names:
            idx = self.device_names.index(self.device_name)
            if self.device_name == "CPU":
                device = torch.device("cpu")
                self.storage.dtype = torch.float32
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
            return upscalers.upscale(latents, UPSCALERS_LATENT[mode], width//8, height//8)

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
            latents = upscalers.upscale(latents, UPSCALERS_LATENT[mode], width//8, height//8)

            if type(latents) == list:
                return torch.stack(latents)
            else:
                return latents
        
        if mode in UPSCALERS_PIXEL:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[mode], width, height)
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)

        self.set_status("Encoding")
        latents = utils.get_latents(vae, seeds, images)
        return latents

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

    def get_prompts(self):
        (prompts, negative_prompts) = self.listify(self.prompt, self.negative_prompt)
        return prompts, negative_prompts
    
    def get_batch_size(self):
        batch_size = max(self.batch_size or 1, 1)
        for i in [self.prompt, self.negative_prompt, self.seeds, self.subseeds, self.image, self.mask]:
            if i != None and type(i) == list:
                batch_size = max(batch_size, len(i))
        return batch_size
    
    def get_metadata(self, mode, width, height, batch_size, prompts, seeds=None, subseeds=None):
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
                if len({self.unet_name, self.clip_name,self.vae_name}) == 1:
                    m["model"] = self.unet_name
                else:
                    m["UNET"] = self.unet_name
                    m["CLIP"] = self.clip_name
                    m["VAE"] = self.vae_name
                m["prompt"] = ' AND '.join(prompts[i][0])
                m["negative_prompt"] = ' AND '.join(prompts[i][1])
                m["seed"] = seeds[i]
                m["steps"] = self.steps
                m["scale"] = format_float(self.scale)
                m["sampler"] = self.sampler
                m["clip_skip"] = self.clip_skip

            if mode == "img2img":
                m["strength"] = format_float(self.strength)
                m["img2img_upscaler"] = self.img2img_upscaler
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            if mode == "txt2img":
                if self.hr_factor and self.hr_factor != 1.0:
                    m["hr_factor"] = format_float(self.hr_factor)
                    m["hr_upscaler"] = self.hr_upscaler
                    m["hr_strength"] = format_float(self.hr_strength)
                    if self.hr_steps and self.hr_steps != self.steps:
                        m["hr_steps"] = self.hr_steps
                    if self.hr_sampler and self.hr_sampler != self.sampler:
                        m["hr_sampler"] = self.hr_sampler
                    if self.hr_eta and self.hr_eta != self.eta:
                        m["hr_eta"] = format_float(self.hr_eta)
            
            if mode == "upscale":
                m["img2img_upscaler"] = self.img2img_upscaler
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            metadata += [m]

        return metadata

    def attach_networks(self, networks):
        self.detach_networks()

        device = self.unet.device

        lora_names = []
        hn_names = []

        for n in networks:
            prefix, n = n.split(":",1)
            if prefix == "lora":
                lora_names += [n]
            if prefix == "hypernet":
                hn_names += [n]

        if lora_names:
            self.set_status("Loading LoRAs")
            self.storage.enforce_network_limit(lora_names, "LoRA")
            self.loras = [self.storage.get_lora(name, device) for name in lora_names]

            for lora in self.loras:
                lora.attach(self.unet.additional, self.clip.additional)
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

    @torch.inference_mode()
    def txt2img(self):
        self.set_status("Loading")
        self.set_device()
        self.load_models()

        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, prompt, width, height, seed, scale, steps".split(", ")
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscaler, hr_strength, hr_sampler, hr_eta".split(", ")
        self.check_parameters(required, optional)

        batch_size = self.get_batch_size()

        device = self.unet.device
        seeds, subseeds = self.get_seeds(batch_size)
        
        self.current_step = 0
        self.total_steps = self.steps

        if self.hr_factor:
            hr_steps = self.hr_steps or self.steps
            self.total_steps += hr_steps
        if not self.hr_sampler:
            self.hr_sampler = self.sampler
        if not self.hr_eta:
            self.hr_eta = self.eta

        metadata = self.get_metadata("txt2img", self.width, self.height, batch_size, self.prompt, seeds, subseeds)

        conditioning = prompts.BatchedConditioningSchedules(self.clip, self.prompt, self.steps, self.clip_skip)
        self.attach_networks(conditioning.get_all_networks())
        conditioning.encode()

        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, self.width // 8, self.height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)
        
        self.set_status("Generating")
        latents = inference.txt2img(denoiser, sampler, noise, self.steps, self.on_step)

        if not self.hr_factor:
            self.set_status("Decoding")
            images = utils.decode_images(self.vae, latents)
            self.on_complete(images, metadata)
            return images

        width = int(self.width * self.hr_factor)
        height = int(self.height * self.hr_factor)

        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)
        conditioning.switch_to_HR(hr_steps)
        self.attach_networks(conditioning.get_all_networks())
        conditioning.encode()

        denoiser.reset()
        sampler = SAMPLER_CLASSES[self.hr_sampler](denoiser, self.hr_eta)

        self.set_status("Upscaling")
        latents = self.upscale_latents(latents, self.hr_upscaler, width, height, seeds)

        self.set_status("Generating")
        latents = inference.img2img(latents, denoiser, sampler, noise, hr_steps, True, self.hr_strength, self.on_step)

        self.set_status("Decoding")
        images = utils.decode_images(self.vae, latents)
        self.on_complete(images, metadata)
        return images

    @torch.inference_mode()
    def img2img(self):
        self.set_status("Loading")
        self.set_device()
        self.load_models()

        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, image, prompt, seed, scale, steps, strength".split(", ")
        optional = "img2img_upscaler, mask, mask_blur, clip_skip, eta, batch_size, padding, width, height".split(", ")
        self.check_parameters(required, optional)

        batch_size = self.get_batch_size()

        device = self.unet.device
        seeds, subseeds = self.get_seeds(batch_size)

        (images,) = self.listify(self.image)
        original_images = images

        if self.mask:
            (masks,)= self.listify(self.mask)

        width, height = images[0].size
        if self.width and self.height:
            width, height = self.width, self.height

        metadata = self.get_metadata("img2img",  width, height, batch_size, self.prompt, seeds, subseeds)

        self.current_step = 0
        self.total_steps = int(self.steps * self.strength) + 1
        
        conditioning = prompts.BatchedConditioningSchedules(self.clip, self.prompt, self.steps, self.clip_skip)
        self.attach_networks(conditioning.get_all_networks())
        conditioning.encode()

        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.mask:
            self.set_status("Preparing")
            images, masks, extents = utils.prepare_inpainting(images, masks, self.padding, width, height)
            masks = upscalers.upscale(masks, transforms.InterpolationMode.NEAREST, width, height)
            masks = [mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]
            mask_latents = utils.get_masks(device, masks)

        self.set_status("Upscaling")
        latents = self.upscale_images(self.vae, images, self.img2img_upscaler, width, height, seeds)
        original_latents = latents

        if self.mask:
            denoiser.set_mask(mask_latents, original_latents)

        self.set_status("Generating")
        latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        self.set_status("Decoding")
        images = utils.decode_images(self.vae, latents)

        if self.mask:
            images = utils.apply_inpainting(images, original_images, masks, extents)

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

        (images,) = self.listify(self.image)
        original_images = images

        if self.mask:
            (masks,)= self.listify(self.mask)

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
            images[0].save("RESULT.png")
            images = utils.apply_inpainting(images, original_images, masks, extents)

        self.on_complete(images, metadata)
        return images
    
    def options(self):
        self.storage.find_all()
        data = {"sampler": list(SAMPLER_CLASSES.keys())}
        for k in self.storage.files:
            data[k] = list(self.storage.files[k].keys())

        data["hr_upscaler"] = list(UPSCALERS_LATENT.keys()) + list(UPSCALERS_PIXEL.keys()) + data["SR"]
        data["img2img_upscaler"] = list(UPSCALERS_PIXEL.keys()) + list(UPSCALERS_LATENT.keys()) + data["SR"]
        data["attention"] = list(CROSS_ATTENTION.keys())
        if not HAVE_XFORMERS:
            data["attention"].remove("xFormers")
        data["TI"] = list(self.storage.embeddings.keys())
        data["device"] = self.device_names

        if self.callback:
            if not self.callback({"type": "options", "data": data}):
                raise RuntimeError("Aborted")
            
    def convert(self):
        self.set_status("Converting")
        convert.autoconvert(self.model_folder, self.trash_folder)
        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")
        
    def build(self):
        self.set_status("Building")
                
        file_type = self.filename.rsplit(".",1)[-1]
        if not file_type  in {"st", "safetensors"}:
            raise ValueError(f"unsuported checkpoint type: {file_type}. supported types are: safetensors, st")

        self.set_status("Finding models")

        source = {}
        unet_file = self.storage.get_filename(self.unet, "UNET")
        source[unet_file] = ["UNET"]

        clip_file = self.storage.get_filename(self.clip, "CLIP")
        source[clip_file] = source.get(clip_file, []) + ["CLIP"]

        vae_file = self.storage.get_filename(self.vae, "VAE")
        source[vae_file] = source.get(vae_file, []) + ["VAE"]

        self.set_status(f"Loading models")
        model = {}
        metadata = {}
        for file in source:
            path = os.path.join(self.storage.path, file)

            state_dict = {}
            if file.rsplit(".",1)[-1] in {"safetensors", "ckpt", "pt"}:
                state_dict = convert.convert_checkpoint(path)
            elif file.endswith(".st"):
                state_dict = safetensors.torch.load_file(path)
            else:
                ValueError(f"unknown format: {file.rsplit(os.path.sep)[-1]}")
            
            source_metadata = {}
            for k in state_dict:
                if k.startswith("metadata."):
                    kk = k.split(".", 1)[1]
                    source_metadata[kk] = ''.join([chr(c) for c in state_dict[k]])
            for k in source_metadata:
                if not k in metadata:
                    metadata[k] = source_metadata[k]
                elif metadata[k] != source_metadata[k]:
                    raise ValueError(f"metadata mismatch: {k}, {metadata[k]} != {source_metadata[k]}")

            model_type = metadata["model_type"]
            for comp in source[file]:
                prefix = f"{model_type}.{comp}."
                found = False
                for k in state_dict.keys():
                    if k.startswith(prefix):
                        found = True
                        model[k] = state_dict[k]
                if not found:
                    raise ValueError(f"model doesnt contain a {model_type} {comp}: {file.rsplit(os.path.sep)[-1]}")
                
            del state_dict

        self.set_status(f"Assembling")

        if not len(list(model.keys())) == {"SDv1":1131, "SDv2":1307}[model_type]:
            raise ValueError(f"model is missing keys")
        
        if file_type == "safetensors":
            model = convert.revert(model_type, model)
        else:
            model["metadata.model_type"] = torch.as_tensor([ord(c) for c in model_type])
            model["metadata.prediction_type"] = torch.as_tensor([ord(c) for c in metadata['prediction_type']])

        self.set_status(f"Saving")
        filename = os.path.join(self.storage.path, "SD", self.filename)
        safetensors.torch.save_file(model, filename)

        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")