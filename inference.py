import torch
import tqdm

def txt2img(denoiser, sampler, noise, steps):
    with torch.autocast(denoiser.unet.autocast(), denoiser.unet.dtype):
        sigmas = sampler.scheduler.get_sigmas(steps)
        latents = noise() * sigmas[0]
        for i in tqdm.trange(steps):
            latents = sampler.step(latents, sigmas, i, noise)
            denoiser.advance(i)
        return latents / 0.18215

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