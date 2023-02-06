import torch
import tqdm
import numpy as np

def txt2img(denoiser, sampler, noise, steps, callback):
    schedule = sampler.scheduler.get_schedule(steps)

    latents = sampler.prepare_noise(noise(), schedule)

    for i in tqdm.trange(steps):
        callback(i, latents)
        denoiser.set_step(i)
        latents = sampler.step(latents, schedule, i, noise)
        
    callback(steps, latents)
    return latents / 0.18215

def img2img(latents, denoiser, sampler, noise, steps, do_exact_steps, strength, callback):
    strength = min(strength, 0.999)
    if do_exact_steps:
        scheduled_steps = int(steps / strength) if strength > 0 else 0
    else:
        scheduled_steps = steps
        steps = int(steps*strength)
    
    schedule = sampler.scheduler.get_truncated_schedule(steps, scheduled_steps)

    latents = latents.to(denoiser.unet.dtype) * 0.18215
    latents = sampler.prepare_latents(latents, noise(), schedule)

    for i in tqdm.trange(steps):
        callback(i, latents)
        denoiser.set_step(i)
        latents = sampler.step(latents, schedule, i, noise)
        
    callback(steps, latents)
    return latents / 0.18215