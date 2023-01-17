import torch
import tqdm
import PIL.Image

import utils

@torch.inference_mode()
def encode_images(vae, seeds, images):
    with torch.autocast(vae.autocast(), vae.dtype):
        images = utils.preprocess_images(images).to(vae.device)
        noise = utils.noise(seeds, images.shape[2] // 8, images.shape[3] // 8, vae.device)

        dists = vae.encode(images).latent_dist
        mean = torch.stack([dists.mean[i%len(images)] for i in range(len(seeds))])
        std = torch.stack([dists.std[i%len(images)] for i in range(len(seeds))])

        latents = mean + std * noise
        
        return latents

@torch.inference_mode()
def decode_images(vae, latents):
    with torch.autocast(vae.autocast(), vae.dtype):
        latents = latents.clone().detach().to(vae.device).to(vae.dtype)
        images = vae.decode(latents).sample
        return utils.postprocess_images(images)

@torch.inference_mode()
def upscale_latents(latents, factor):
    latents = torch.nn.functional.interpolate(latents.clone().detach(), scale_factor=(factor,factor), mode="bilinear", antialias=False)
    return latents

@torch.inference_mode()
def get_latents(vae, seeds, images):
    if type(images) == torch.Tensor:
        return images.to(vae.device)
    elif type(images) == list:
        return encode_images(vae, seeds, images)

@torch.inference_mode()
def get_masks(device, masks):
    if type(masks) == torch.Tensor:
        return masks      
    elif type(masks) == list:
        return utils.preprocess_masks(masks).to(device)

@torch.inference_mode()
def apply_masks(images, originals, masks):
    for i in range(len(images)):
        m = masks[i%len(masks)]
        o = originals[i%len(originals)]
        images[i] = PIL.Image.composite(images[i], o, m)
    return images

@torch.inference_mode()
def txt2img(denoiser, sampler, noise, steps):
    with torch.autocast(denoiser.unet.autocast(), denoiser.unet.dtype):
        sigmas = sampler.scheduler.get_sigmas(steps)
        latents = noise() * sigmas[0]
        for i in tqdm.trange(steps):
            latents = sampler.step(latents, sigmas, i, noise)
            denoiser.advance(i)
        return latents / 0.18215

@torch.inference_mode()
def img2img(latents, denoiser, sampler, noise, steps, do_exact_steps, strength):
    with torch.autocast(denoiser.unet.autocast(), denoiser.unet.dtype):
        strength = min(strength, 0.999)
        if do_exact_steps:
            requested_steps = steps
            steps = int(requested_steps / strength) if strength > 0 else 0
            skipped_steps = steps - requested_steps
        else:
            skipped_steps = steps - int(steps*strength) - 1

        sigmas = sampler.scheduler.get_sigmas(steps)[skipped_steps:]
        steps = len(sigmas)-1

        latents = latents * 0.18215
        latents = latents + noise() * sigmas[0]

        for i in tqdm.trange(steps):
            latents = sampler.step(latents, sigmas, i, noise)
            denoiser.advance(i)
        return latents / 0.18215

