import torch

import prompts
import samplers
import utils
import inference

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler_a", "clip_skip": 1, "eta": 1, "do_exact_steps": False,
    "hr_upscale": "latent", "hr_strength": 0.7
}

SAMPLER_CLASSES = {
    "Euler": samplers.Euler,
    "Euler a": samplers.Euler_a,
    "DPM++ 2M": samplers.DPM_2M,
    "DPM++ 2S a": samplers.DPM_2S_a,
    "DPM++ SDE": samplers.DPM_SDE
}

class GenerationParameters():
    def __init__(self, storage: utils.ModelStorage, device):
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
        args = [[a] if type(a) != list else a for a in args]
        mx = max([len(a) for a in args])
        return args, mx

    @torch.inference_mode()
    def txt2img(self):
        self.load_models()

        required = "unet, clip, vae, sampler, prompt, negative_prompt, width, height, seed, scale, steps".split(", ")
        optional = "clip_skip, eta, batch_size, hr_steps, hr_factor, hr_upscale, hr_strength".split(", ")

        self.check_parameters(required, optional)

        (positive_prompts, negative_prompts, seeds), batch_size = self.listify(self.prompt, self.negative_prompt, self.seed)

        if self.batch_size:
            batch_size = max(batch_size, self.batch_size)

        if len(seeds) < batch_size:
            seeds += [i + seeds[-1] + 1 for i in range(batch_size-len(seeds))]

        device = self.unet.device

        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = samplers.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, self.width // 8, self.height // 8, device)

        sampler_class = SAMPLER_CLASSES[self.sampler]
        sampler = sampler_class(denoiser, self.eta)

        latents = inference.txt2img(denoiser, sampler, noise, self.steps)

        if not self.hr_factor:
            return inference.decode_images(self.vae, latents)

        width = self.width * self.hr_factor
        height = self.width * self.hr_factor

        noise = utils.NoiseSchedule(seeds, width // 8, height // 8, device)
        conditioning.reset()
        denoiser.reset()
        sampler.reset()

        latents = inference.upscale_latents(latents, self.hr_factor)
        hr_steps = self.hr_steps or self.steps

        latents = inference.img2img(latents, denoiser, sampler, noise, hr_steps, True, self.hr_strength)

        return inference.decode_images(self.vae, latents)

    @torch.inference_mode()
    def img2img(self):
        self.load_models()

        required = "unet, clip, vae, sampler, image, prompt, negative_prompt, seed, scale, steps, strength".split(", ")
        optional = "mask, clip_skip, eta, do_exact_steps, batch_size".split(", ")

        self.check_parameters(required, optional)

        (images, positive_prompts, negative_prompts, seeds), batch_size = self.listify(self.image, self.prompt, self.negative_prompt, self.seed)

        if self.mask:
            (masks,), mx = self.listify(self.mask)
            batch_size = max(batch_size, mx)

        if self.batch_size:
            batch_size = max(batch_size, self.batch_size)

        if len(seeds) < batch_size:
            seeds += [i + seeds[-1] + 1 for i in range(batch_size-len(seeds))]

        device = self.unet.device

        conditioning = prompts.ConditioningSchedule(self.clip, positive_prompts, negative_prompts, self.steps, self.clip_skip, batch_size)
        denoiser = samplers.GuidedDenoiser(self.unet, conditioning, self.scale)
        noise = utils.NoiseSchedule(seeds, self.width // 8, self.height // 8, device)

        sampler_class = SAMPLER_CLASSES[self.sampler]
        sampler = sampler_class(denoiser, self.eta)
        originals = images

        original_latents = inference.get_latents(self.vae, seeds, originals)

        if self.mask:
            mask_latents = inference.get_masks(device, masks)
            denoiser.set_mask(mask_latents, original_latents)

        latents = inference.img2img(original_latents, denoiser, sampler, noise, self.steps, self.do_exact_steps, self.strength)

        images = inference.decode_images(self.vae, latents)

        if self.mask:
            images = inference.apply_masks(images, originals, masks)
        
        return images



