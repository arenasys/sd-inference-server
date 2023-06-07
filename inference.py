import torch
import tqdm
import numpy as np

@torch.cuda.amp.autocast()
def txt2img(denoiser, sampler, noise, steps, callback):
    schedule = sampler.scheduler.get_schedule(steps)

    latents = sampler.prepare_noise(noise(), schedule)

    iter = tqdm.trange(steps)
    for i in iter:
        denoiser.set_step(i)
        latents = sampler.step(latents, schedule, i, noise)
        callback(iter.format_dict, denoiser.predictions)
    return latents / 0.18215

@torch.cuda.amp.autocast()
def img2img(latents, denoiser, sampler, noise, steps, do_exact_steps, strength, callback):
    strength = min(strength, 0.999)
    if do_exact_steps:
        scheduled_steps = int(steps / strength) if strength > 0 else 0
    else:
        scheduled_steps = steps
        steps = int(steps*strength)
    
    schedule = sampler.scheduler.get_truncated_schedule(steps, scheduled_steps)

    latents = latents.to(denoiser.unet.dtype) * 0.18215

    if scheduled_steps != 0:
        latents = sampler.prepare_latents(latents, noise(), schedule)
        iter = tqdm.trange(steps)
        for i in iter:
            denoiser.set_step(i)
            latents = sampler.step(latents, schedule, i, noise)
            callback(iter.format_dict, denoiser.predictions)
    return latents / 0.18215