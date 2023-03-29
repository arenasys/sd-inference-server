import torch
import torchvision.transforms as transforms
import PIL
import random
import io

import prompts
import samplers_k
import samplers_ddpm
import guidance
import utils
import storage
import upscalers
import inference
import convert

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler_a", "clip_skip": 1, "eta": 1,
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
    "Latent (lanczos)": transforms.InterpolationMode.LANCZOS,
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

class GenerationParameters():
    def __init__(self, storage: storage.ModelStorage, device):
        self.storage = storage
        self.device = device

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
            if not attr in ["storage", "device", "callback"]:
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
                        if type(value[i]) == bytes or type(value) == bytearray:
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
            return torch.stack(latents)
        
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
            (subseeds,) = self.listify(self.subseed)
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
    
    def get_metadata(self, mode, batch_size, prompts, negative_prompts, seeds, subseeds, width, height):
        metadata = []
        for i in range(batch_size):
            m = {
                "mode": mode,
                "prompt": prompts[i%len(prompts)],
                "negative_prompt": negative_prompts[i%len(negative_prompts)],
                "seed": seeds[i],
                "width": width,
                "height": height,
                "steps": self.steps,
                "scale": self.scale,            
                "sampler": self.sampler,
                "clip_skip": self.clip_skip
            }

            if subseeds != None:
                sds = []
                strs = []
                active = False
                for s, r in subseeds:
                    if r != 0.0:
                        active = True
                    sds += [str(s)]
                    strs += [str(r)]
                if active:
                    m["subseed"] = ", ".join(sds)
                    m["subseed_strength"] = ", ".join(strs)

            if self.eta != DEFAULTS["eta"]:
                m["eta"] = self.eta

            if len({self.unet_name, self.clip_name,self.vae_name}) == 1:
                m["model"] = self.unet_name
            else:
                m["UNET"] = self.unet_name
                m["CLIP"] = self.clip_name
                m["VAE"] = self.vae_name

            if mode == "img2img":
                m["img2img_upscaler"] = self.img2img_upscaler
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            if mode == "txt2img":
                if self.hr_factor and self.hr_factor != 1.0:
                    m["hr_steps"] = self.hr_steps
                    m["hr_factor"] = self.hr_factor
                    m["hr_upscaler"] = self.hr_upscaler
                    m["hr_strength"] = self.hr_strength
                    m["hr_sampler"] = self.hr_sampler
                    m["hr_eta"] = self.hr_eta

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
        self.load_models()

        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, prompt, negative_prompt, width, height, seed, scale, steps".split(", ")
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscaler, hr_strength, hr_sampler, hr_eta".split(", ")
        self.check_parameters(required, optional)

        batch_size = self.get_batch_size()

        device = self.unet.device
        positive_prompts, negative_prompts = self.get_prompts()
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

        metadata = self.get_metadata("txt2img", batch_size, positive_prompts, negative_prompts, seeds, subseeds, self.width, self.height)

        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
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
        conditioning.switch_to_HR()
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
        self.load_models()

        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, image, prompt, negative_prompt, seed, scale, steps, strength".split(", ")
        optional = "img2img_upscaler, mask, mask_blur, clip_skip, eta, batch_size, padding, width, height".split(", ")
        self.check_parameters(required, optional)

        batch_size = self.get_batch_size()

        device = self.unet.device
        positive_prompts, negative_prompts = self.get_prompts()
        seeds, subseeds = self.get_seeds(batch_size)

        (images,) = self.listify(self.image)
        original_images = images

        if self.mask:
            (masks,)= self.listify(self.mask)

        width, height = images[0].size
        if self.width and self.height:
            width, height = self.width, self.height

        metadata = self.get_metadata("img2img", batch_size, positive_prompts, negative_prompts, seeds, subseeds, width, height)

        self.current_step = 0
        self.total_steps = int(self.steps * self.strength) + 1
        
        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        self.attach_networks(conditioning.get_all_networks())
        conditioning.encode()

        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.mask:
            self.set_status("Preparing")
            images, masks, extents = utils.prepare_inpainting(images, masks, self.padding, width, height)

        self.set_status("Upscaling")
        latents = self.upscale_images(self.vae, images, self.img2img_upscaler, width, height, seeds)
        original_latents = latents

        if self.mask:
            masks = upscalers.upscale(masks, transforms.InterpolationMode.NEAREST, width, height)
            masks = [mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]
            mask_latents = utils.get_masks(device, masks)
            denoiser.set_mask(mask_latents, original_latents)

        self.set_status("Generating")
        latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        self.set_status("Decoding")
        images = utils.decode_images(self.vae, latents)

        if self.mask:
            images = utils.apply_inpainting(images, original_images, masks, extents)

        self.on_complete(images, metadata)
        return images
    
    def options(self):
        self.storage.find_all()
        data = {"sampler": list(SAMPLER_CLASSES.keys())}
        for k in self.storage.files:
            data[k] = list(self.storage.files[k].keys())

        data["upscaler"] = list(UPSCALERS_LATENT.keys()) + list(UPSCALERS_PIXEL.keys()) 

        data["SR"] = data["SR"]

        if self.callback:
            if not self.callback({"type": "options", "data": data}):
                raise RuntimeError("Aborted")
            
    def convert(self):
        self.set_status("Converting")
        convert.autoconvert(self.model_folder, self.trash_folder)
        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")