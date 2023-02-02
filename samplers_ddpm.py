import torch
import numpy as np

class DDPMScheduler():
    def __init__(self):
        self.alphas = self.get_alphas()

    def get_schedule(self, steps):
        timesteps = np.flip(np.arange(1, 1000, 1000//steps))
        return timesteps

    def get_truncated_schedule(self, steps, scheduled_steps):
        timesteps = self.get_schedule(scheduled_steps)
        return timesteps[-steps:]

    def get_alphas(self):
        betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        return alphas_cumprod

    def to(self, dtype, device):
        self.alphas = self.alphas.to(dtype).to(device)
        return self

class DDPMSampler():
    def __init__(self, model, scheduler, eta):
        self.model = model
        self.scheduler = scheduler or DDPMScheduler().to(model.dtype, model.device)
        self.eta = eta

    def predict(self, latents, timestep):
        return self.model.predict_noise(latents, timestep)

    def prepare_noise(self, noise, timesteps):
        return noise

    def prepare_latents(self, latents, noise, timesteps):
        alpha = self.scheduler.alphas[timesteps[0]]
        return alpha.sqrt() * latents + (1-alpha).sqrt() * noise

    def reset(self):
        pass

class DDIM(DDPMSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.eta = 1.0

    def step(self, x, timesteps, i, noise):
        t =  timesteps[i]
        
        if i+1 >= len(timesteps):
            t_prev = timesteps[-1]
            a_prev = torch.tensor(1)
        else:
            t_prev = timesteps[i+1]
            a_prev = self.scheduler.alphas[t_prev]

        x = self.model.mask_noise(x, a_prev, noise())

        e_t = self.predict(x, t)
        a_t = self.scheduler.alphas[t]
        
        pred_x0 = (x - (1-a_t).sqrt() * e_t) / a_t.sqrt()

        dir_xt = (1.0 - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev