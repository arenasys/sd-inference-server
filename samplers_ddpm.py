import torch
import numpy as np

class DDPMScheduler():
    def __init__(self):
        self.alphas = self.get_alphas()

    def get_schedule(self, steps):
        timesteps = np.flip(np.arange(1, 1000, 1000//steps)).copy()
        return timesteps

    def get_truncated_schedule(self, steps, scheduled_steps):
        timesteps = self.get_schedule(scheduled_steps)
        return timesteps[-steps:]

    def get_alphas(self):
        betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000) ** 2
        alphas = torch.cumprod(1.0 - betas, 0)
        return alphas

    def to(self, dtype, device):
        # fp32 required for DDIM eta
        self.alphas = self.alphas.to(device)
        return self

class DDPMSampler():
    def __init__(self, model, scheduler, eta):
        self.model = model
        self.scheduler = scheduler or DDPMScheduler().to(model.dtype, model.device)
        self.eta = eta

    def predict(self, latents, timestep, alpha):
        return self.model.predict_noise(latents, timestep, alpha)

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
        self.eta = 0.0

    def step(self, x, timesteps, i, noise):        
        t = timesteps[i]
        a_t = self.scheduler.alphas[t]

        if i+1 >= len(timesteps):
            a_prev = torch.tensor(0.9990)
        else:
            a_prev = self.scheduler.alphas[timesteps[i+1]]

        x = self.model.mask_noise(x, a_prev, noise)
        e_t = self.predict(x, t, a_t)

        sigma_t = 0.0
        if self.eta:
            sigma_t = self.eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()

        pred_x0 = (x - (1-a_t).sqrt() * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise()
        return x_prev