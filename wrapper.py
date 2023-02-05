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

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler_a", "clip_skip": 1, "eta": 1,
    "hr_upscale": "Latent (nearest)", "hr_strength": 0.7, "img2img_upscale": "Lanczos", "mask_blur": 4,
    "lora_strength": 1.0, "hn_strength": 1.0
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

    def set_progress(self, current, total):
        if self.callback:
            if not self.callback({"type": "progress", "data": {"current": current, "total": total}}):
                raise RuntimeError("Aborted")

    def on_step(self, step, latents):
        if step:
            self.current_step += 1
        self.set_progress(self.current_step, self.total_steps)
    
    def on_complete(self, images):
        if self.callback:
            images_data = []
            for i in images:
                bytesio = io.BytesIO()
                i.save(bytesio, format='PNG')
                images_data += [bytesio.getvalue()]
            self.callback({"type": "result", "data": {"images": images_data}})

    def reset(self):
        for attr in list(self.__dict__.keys()):
            if not attr in ["storage", "device", "callback"]:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if type(value) == bytes:
                value = PIL.Image.open(io.BytesIO(value))
            setattr(self, key, value)
    
    def load_models(self):
        if not self.unet or type(self.unet) == str:
            self.set_status("Loading UNET")
            self.unet = self.storage.get_unet(self.unet or self.model, self.device)
        
        if not self.clip or type(self.clip) == str:
            self.set_status("Loading CLIP")
            self.clip = self.storage.get_clip(self.clip or self.model, self.device)
            self.clip.set_textual_inversions(self.storage.get_embeddings(self.device))
        
        if not self.vae or type(self.vae) == str:
            self.set_status("Loading VAE")
            self.vae = self.storage.get_vae(self.vae or self.model, self.device)

        self.storage.clear_file_cache()

        if self.hr_upscale:
            if not self.hr_upscale in UPSCALERS_LATENT and not self.hr_upscale in UPSCALERS_PIXEL:
                self.set_status("status", "Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.hr_upscale, self.device)

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

        if missing:
            raise ValueError(f"ERROR missing required parameters: {', '.join(missing)}")

        if not self.sampler in SAMPLER_CLASSES:
            raise ValueError(f"ERROR unknown sampler: {self.sampler}")

        if (self.width or self.height) and not (self.width and self.height):
            raise ValueError("ERROR width and height must both be set")

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

        latents = utils.get_latents(vae, seeds, images)
        return latents

    def get_seeds(self, batch_size):
        (seeds,) = self.listify(self.seed)
        if self.subseed:
            (subseeds,) = self.listify(self.subseed)
        else:
            subseeds = [(0,0)]

        batch_size = max(batch_size, len(seeds), len(subseeds))

        for i in range(len(seeds)):
            if seeds[i] == -1:
                seeds[i] = random.randrange(4294967294)
        for i in range(len(subseeds)):
            if subseeds[i][0] == -1:
                subseeds[i] = (random.randrange(4294967294), subseeds[i][1])

        if len(seeds) < batch_size:
            last_seed = seeds[-1]
            seeds += [last_seed + i + 1 for i in range(batch_size-len(seeds))]

        if len(subseeds) < batch_size:
            last_seed, last_strength = subseeds[-1]
            subseeds += [(last_seed + i + 1, last_strength) for i in range(batch_size-len(subseeds))]

        return seeds, subseeds, batch_size

    def get_prompts(self, batch_size):
        (prompts, negative_prompts) = self.listify(self.prompt, self.negative_prompt)
        batch_size = max(batch_size or 0, len(prompts), len(negative_prompts))
        return prompts, negative_prompts, batch_size

    def attach_networks(self):
        self.detach_networks()

        device = self.unet.device

        if self.lora:
            self.set_status("Loading LoRAs")
            (lora_names, lora_strengths) = self.listify(self.lora, self.lora_strength)

            self.storage.enforce_network_limit(lora_names, "LoRA")
            self.loras = [self.storage.get_lora(name, device) for name in lora_names]

            if len(lora_strengths) < len(self.loras):
                lora_strengths += [1 for _ in range(len(self.loras)-len(lora_strengths))]

            for i, lora in enumerate(self.loras):
                lora.set_strength(lora_strengths[i])
                lora.attach(self.unet.additional, self.clip.additional)
        else:
            self.storage.enforce_network_limit([], "LoRA")

        if self.hn:
            self.set_status("Loading Hypernetworks")
            (hn_names, hn_strengths) = self.listify(self.hn, self.hn_strength)

            self.storage.enforce_network_limit(hn_names, "HN")
            self.hns = [self.storage.get_hypernetwork(name, device) for name in hn_names]

            if len(hn_strengths) < len(self.hns):
                hn_strengths += [1 for _ in range(len(self.hns)-len(hn_strengths))]

            for i, hn in enumerate(self.hns):
                hn.set_strength(hn_strengths[i])
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
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscale, hr_strength, hr_sampler, hr_eta, lora, hn".split(", ")
        self.check_parameters(required, optional)

        device = self.unet.device
        positive_prompts, negative_prompts, batch_size = self.get_prompts(self.batch_size)
        seeds, subseeds, batch_size = self.get_seeds(batch_size)
        
        self.attach_networks()

        self.current_step = 0
        self.total_steps = self.steps

        if self.hr_factor:
            hr_steps = self.hr_steps or self.steps
            self.total_steps += hr_steps
        if not self.hr_sampler:
            self.hr_sampler = self.sampler
        if not self.hr_eta:
            self.hr_eta = self.eta

        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, self.width // 8, self.height // 8, device)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)
        
        self.set_status("Generating")
        latents = inference.txt2img(denoiser, sampler, noise, self.steps, self.on_step)

        if not self.hr_factor:
            images = utils.decode_images(self.vae, latents)
            self.on_complete(images)
            return images

        width = int(self.width * self.hr_factor)
        height = int(self.height * self.hr_factor)

        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device)
        conditioning.switch_to_HR()
        denoiser.reset()
        sampler = SAMPLER_CLASSES[self.hr_sampler](denoiser, self.hr_eta)

        self.set_status("Upscaling")
        latents = self.upscale_latents(latents, self.hr_upscale, width, height, seeds)

        self.set_status("Generating")
        latents = inference.img2img(latents, denoiser, sampler, noise, hr_steps, True, self.hr_strength, self.on_step)

        images = utils.decode_images(self.vae, latents)
        self.on_complete(images)
        return images

    @torch.inference_mode()
    def img2img(self):
        self.set_status("Loading")
        self.load_models()

        self.set_status("Configuring")
        required = "unet, clip, vae, sampler, image, prompt, negative_prompt, seed, scale, steps, strength".split(", ")
        optional = "mask, mask_blur, clip_skip, eta, batch_size, padding, width, height, lora".split(", ")
        self.check_parameters(required, optional)

        device = self.unet.device
        positive_prompts, negative_prompts, batch_size = self.get_prompts(self.batch_size)
        seeds, subseeds, batch_size = self.get_seeds(batch_size)

        (images,) = self.listify(self.image)
        batch_size = max(batch_size, len(images))
        original_images = images

        if self.mask:
            (masks,)= self.listify(self.mask)
            batch_size = max(batch_size, len(masks))

        width, height = images[0].size
        if self.width and self.height:
            width, height = self.width, self.height

        self.attach_networks()

        self.current_step = 0
        self.total_steps = int(self.steps * self.strength) + 1
            
        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = guidance.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.mask:
            images, masks, extents = utils.prepare_inpainting(images, masks, self.padding)

        self.set_status("Upscaling")
        latents = self.upscale_images(self.vae, images, self.img2img_upscale, width, height, seeds)
        original_latents = latents

        if self.mask:
            masks = upscalers.upscale(masks, transforms.InterpolationMode.NEAREST, width, height)
            masks = [mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]
            mask_latents = utils.get_masks(device, masks)
            denoiser.set_mask(mask_latents, original_latents)

        self.set_status("Generating")
        latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        images = utils.decode_images(self.vae, latents)

        if self.mask:
            images = utils.apply_inpainting(images, original_images, masks, extents)

        self.on_complete(images)
        return images


