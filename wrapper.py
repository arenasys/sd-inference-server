import torch
import torchvision.transforms as transforms
import PIL
import random

import prompts
import samplers
import utils
import storage
import upscalers
import inference

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler_a", "clip_skip": 1, "eta": 1, "do_exact_steps": False,
    "hr_upscale": "Latent (nearest)", "hr_strength": 0.7, "img2img_upscale": "Lanczos", "mask_blur": 4
}

SAMPLER_CLASSES = {
    "Euler": samplers.Euler,
    "Euler a": samplers.Euler_a,
    "DPM++ 2M": samplers.DPM_2M,
    "DPM++ 2S a": samplers.DPM_2S_a,
    "DPM++ SDE": samplers.DPM_SDE
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

    def reset(self):
        for attr in list(self.__dict__.keys()):
            if not attr in ["storage", "device"]:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def load_models(self):
        if not self.unet or type(self.unet) == str:
            self.unet = self.storage.get_unet(self.unet or self.model, self.device)
        
        if not self.clip or type(self.clip) == str:
            self.clip = self.storage.get_clip(self.clip or self.model, self.device)
        
        if not self.vae or type(self.vae) == str:
            self.vae = self.storage.get_vae(self.vae or self.model, self.device)

        self.storage.clear_cache()

        if self.hr_upscale:
            if not self.hr_upscale in UPSCALERS_LATENT and not self.hr_upscale in UPSCALERS_PIXEL:
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
        
        if (self.width or self.height) and not (self.width and self.height):
            raise ValueError("ERROR width and height must both be set")

        return seeds, subseeds, batch_size

    def get_prompts(self, batch_size):
        (prompts, negative_prompts) = self.listify(self.prompt, self.negative_prompt)
        batch_size = max(batch_size or 0, len(prompts), len(negative_prompts))
        return prompts, negative_prompts, batch_size

    @torch.inference_mode()
    def txt2img(self):
        self.load_models()
        required = "unet, clip, vae, sampler, prompt, negative_prompt, width, height, seed, scale, steps".split(", ")
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscale, hr_strength".split(", ")
        self.check_parameters(required, optional)

        device = self.unet.device
        positive_prompts, negative_prompts, batch_size = self.get_prompts(self.batch_size)
        seeds, subseeds, batch_size = self.get_seeds(batch_size)

        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = samplers.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, self.width // 8, self.height // 8, device)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        latents = inference.txt2img(denoiser, sampler, noise, self.steps)

        if not self.hr_factor:
            return utils.decode_images(self.vae, latents)

        width = int(self.width * self.hr_factor)
        height = int(self.height * self.hr_factor)

        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device)
        conditioning.reset()
        denoiser.reset()
        sampler.reset()

        latents = self.upscale_latents(latents, self.hr_upscale, width, height, seeds)

        hr_steps = self.hr_steps or self.steps

        latents = inference.img2img(latents, denoiser, sampler, noise, hr_steps, True, self.hr_strength)

        return utils.decode_images(self.vae, latents)

    @torch.inference_mode()
    def img2img(self):
        self.load_models()
        required = "unet, clip, vae, sampler, image, prompt, negative_prompt, seed, scale, steps, strength".split(", ")
        optional = "mask, mask_blur, clip_skip, eta, do_exact_steps, batch_size, padding, width, height".split(", ")
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
            
        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = samplers.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.mask:
            images, masks, extents = utils.prepare_inpainting(images, masks, self.padding)

        latents = self.upscale_images(self.vae, images, self.img2img_upscale, width, height, seeds)
        original_latents = latents

        if self.mask:
            masks = upscalers.upscale(masks, transforms.InterpolationMode.NEAREST, width, height)
            masks = [mask.filter(PIL.ImageFilter.GaussianBlur(self.mask_blur)) for mask in masks]
            mask_latents = utils.get_masks(device, masks)
            denoiser.set_mask(mask_latents, original_latents)

        latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, self.do_exact_steps, self.strength)

        images = utils.decode_images(self.vae, latents)

        if self.mask:
            images = utils.apply_inpainting(images, original_images, masks, extents)
        
        return images



