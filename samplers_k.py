import torch
import numpy as np
from k_diffusion.sampling import get_ancestral_step, to_d, BrownianTreeNoiseSampler, append_zero
from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential

MN, MX = 0.0312652550637722, 14.611639022827148

class KScheduler():
    def __init__(self):
        self.sigmas, self.log_sigmas = self.get_sigmas()
        self.dtype = torch.float32

    def get_schedule(self, steps):
        timesteps = torch.linspace(999, 0, steps, device=self.sigmas.device).float()
        low_idx, high_idx, w = timesteps.floor().long(), timesteps.ceil().long(), timesteps.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        sigmas = log_sigma.exp()
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return sigmas.to(self.dtype)

    def get_truncated_schedule(self, steps, scheduled_steps):
        timesteps = self.get_schedule(scheduled_steps+1)
        return timesteps[-(steps + 1):].to(self.dtype)

    def get_sigmas(self):
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
        alphas = torch.tensor(np.cumprod(1. - betas.numpy(), axis=0), dtype=torch.float16)
        sigmas = ((1 - alphas) / alphas).to(torch.float32) ** 0.5
        log_sigmas = sigmas.log()
        return sigmas, log_sigmas
    
    def rescale_to_znsr(self):
        dtype = self.dtype
        device = self.sigmas.device

        sigmas, _ = self.get_sigmas()

        alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        alphas_bar_sqrt -= (alphas_bar_sqrt_T)

        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas_bar[-1] = 4.8973451890853435e-08
        
        self.sigmas = ((1 - alphas_bar) / alphas_bar) ** 0.5
        self.log_sigmas = self.sigmas.log()

        self.to(device, dtype)

    def to(self, device, dtype):
        self.dtype = dtype
        self.sigmas = self.sigmas.to(device)
        self.log_sigmas = self.log_sigmas.to(device)
        return self

    def sigma_to_timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def timestep_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

class KSampler():
    def __init__(self, model, scheduler, eta):
        self.model = model
        self.scheduler = scheduler or KScheduler().to(model.device, model.dtype)
        self.eta = eta

    def predict(self, latents, sigma):
        timestep = self.scheduler.sigma_to_timestep(sigma)
        original = self.model.predict_original(latents, timestep, sigma)
        self.model.set_predictions(original)
        return original

    def prepare_noise(self, noise, sigmas):
        return noise * sigmas[0]
    
    def prepare_latents(self, latents, noise, sigmas):
        return latents + noise * sigmas[0]

    def reset(self):
        pass

class DPM_SDE(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.reset()

    def reset(self):
        self.noise_samplers = []
        self.noise_states = []

    def initialize_noise(self, x, sigma_min, sigma_max, noise):
        seeds = noise.seeds
        shape = x[0][None,:]

        for i in range(len(seeds)):
            torch.manual_seed(seeds[i])
            noise_sampler = BrownianTreeNoiseSampler(shape, sigma_min, sigma_max)
            self.noise_samplers += [noise_sampler]
            self.noise_states += [torch.get_rng_state()]

    def sample_noise(self, t0, t1):
        noises = []
        for i in range(len(self.noise_samplers)):
            torch.set_rng_state(self.noise_states[i])
            noise = self.noise_samplers[i](t0, t1)
            noises += [noise]
            self.noise_states[i] = torch.get_rng_state()
        
        noises = torch.cat(noises)
        return noises

    def step(self, x, sigmas, i, noise):
        """DPM-Solver++ (stochastic)."""

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        if not self.noise_samplers:
            self.initialize_noise(x, sigma_min, sigma_max, noise)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        r = 1 / 2

        denoised = self.predict(x, sigmas[i])
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), self.eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + self.sample_noise(sigma_fn(t), sigma_fn(s)) * su
            denoised_2 = self.predict(x_2, sigma_fn(s))

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), self.eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + self.sample_noise(sigma_fn(t), sigma_fn(t_next)) * su
            
        return x

class DPM_2M(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.reset()

    def reset(self):
        self.prev_denoised = None

    def step(self, x, sigmas, i, noise):
        """DPM-Solver++(2M)."""

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        denoised = self.predict(x, sigmas[i])
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t

        if self.prev_denoised == None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * self.prev_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        self.prev_denoised = denoised

        return x

class DPM_2S_a(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
    
    def step(self, x, sigmas, i, noise):
        """Ancestral sampling with DPM-Solver++(2S) second-order steps."""

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        denoised = self.predict(x, sigmas[i])
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=self.eta)
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = self.predict(x_2, sigma_fn(s))
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise() * sigma_up
        return x
    
class DPM_2M_SDE(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.solver_type = 'midpoint'
        self.old_denoised = None
        self.h_last = None
        self.reset()

    def reset(self):
        self.noise_samplers = []
        self.noise_states = []

    def initialize_noise(self, x, sigma_min, sigma_max, noise):
        seeds = noise.seeds
        shape = x[0][None,:]

        for i in range(len(seeds)):
            torch.manual_seed(seeds[i])
            noise_sampler = BrownianTreeNoiseSampler(shape, sigma_min, sigma_max)
            self.noise_samplers += [noise_sampler]
            self.noise_states += [torch.get_rng_state()]

    def sample_noise(self, t0, t1):
        noises = []
        for i in range(len(self.noise_samplers)):
            torch.set_rng_state(self.noise_states[i])
            noise = self.noise_samplers[i](t0, t1)
            noises += [noise]
            self.noise_states[i] = torch.get_rng_state()
        
        noises = torch.cat(noises)
        return noises

    def step(self, x, sigmas, i, noise):
        """DPM-Solver++(2M) SDE."""

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        if not self.noise_samplers:
            self.initialize_noise(x, sigma_min, sigma_max, noise)

        denoised = self.predict(x, sigmas[i])

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = self.eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if self.old_denoised is not None:
                r = self.h_last / h
                if self.solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - self.old_denoised)
                elif self.solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - self.old_denoised)

            if self.eta:
                x = x + self.sample_noise(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt()# * s_noise

            self.h_last = h

        self.old_denoised = denoised
        return x

class DPM_3M_SDE(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.old_denoised_1, self.old_denoised_2 = None, None
        self.h_last_1, self.h_last_2 = None, None
        self.reset()

    def reset(self):
        self.noise_samplers = []
        self.noise_states = []

    def initialize_noise(self, x, sigma_min, sigma_max, noise):
        seeds = noise.seeds
        shape = x[0][None,:]

        for i in range(len(seeds)):
            torch.manual_seed(seeds[i])
            noise_sampler = BrownianTreeNoiseSampler(shape, sigma_min, sigma_max)
            self.noise_samplers += [noise_sampler]
            self.noise_states += [torch.get_rng_state()]

    def sample_noise(self, t0, t1):
        noises = []
        for i in range(len(self.noise_samplers)):
            torch.set_rng_state(self.noise_states[i])
            noise = self.noise_samplers[i](t0, t1)
            noises += [noise]
            self.noise_states[i] = torch.get_rng_state()
        
        noises = torch.cat(noises)
        return noises
    
    def step(self, x, sigmas, i, noise):
        """DPM-Solver++(3M) SDE."""

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        if not self.noise_samplers:
            self.initialize_noise(x, sigma_min, sigma_max, noise)

        denoised_1, denoised_2 = self.old_denoised_1, self.old_denoised_2
        h_1, h_2 = self.h_last_1, self.h_last_2

        denoised = self.predict(x, sigmas[i])

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (self.eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if self.eta:
                x = x + self.sample_noise(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * self.eta).expm1().neg().sqrt()# * s_noise

            h_1, h_2 = h, h_1
            self.h_last_1, self.h_last_2 = h_1, h_2

        denoised_1, denoised_2 = denoised, denoised_1
        self.old_denoised_1, self.old_denoised_2 = denoised_1, denoised_2
        return x

class Euler(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)

    def step(self, x, sigmas, i, noise):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""

        s_churn, s_tmin, s_tmax = 0.0, 0.0, float('inf')
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.

        eps = noise()
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = self.predict(x, sigma_hat)
        d = to_d(x, sigma_hat, denoised)

        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        return x

class Euler_a(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.eta = 1.0

    def step(self, x, sigmas, i, noise):
        """Ancestral sampling with Euler method steps."""

        denoised = self.predict(x, sigmas[i])
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=self.eta)
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        noise = noise()
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise * sigma_up

        return x
    
class Euler_a_CFG_PP(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)
        self.eta = 1.0
        self.model.set_cfg_pp(True)

    def step(self, x, sigmas, i, noise):
        """Ancestral sampling with Euler method steps."""

        denoised = self.predict(x, sigmas[i])
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=self.eta)
        d = to_d(x, sigmas[i], self.model.uncond_pred)
        # Euler method
        noise = noise()
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise * sigma_up
        
        return x
class LCM(KSampler):
    def __init__(self, model, eta=1.0, scheduler=None):
        super().__init__(model, scheduler, eta)

    def step(self, x, sigmas, i, noise):
        x = self.predict(x, sigmas[i])
        if sigmas[i + 1] > 0:
            x = x + (sigmas[i + 1] * noise())
        return x
    
class SchedulerKarras(KScheduler):
    def get_schedule(self, steps):
        sigmas = get_sigmas_karras(n=steps, sigma_min=MN, sigma_max=MX, device=self.sigmas.device).to(self.dtype)
        return sigmas
    
class SchedulerExponential(KScheduler):
    def get_schedule(self, steps):
        sigmas = get_sigmas_exponential(n=steps, sigma_min=MN, sigma_max=MX, device=self.sigmas.device).to(self.dtype)
        return sigmas
    
class SchedulerUniform(KScheduler):
    def get_schedule(self, steps):
        start = self.sigma_to_timestep(torch.tensor(MX))
        end = self.sigma_to_timestep(torch.tensor(MN))
        sigs = [
            self.timestep_to_sigma(ts)
            for ts in torch.linspace(start, end, steps+1)[:-1]
        ]
        sigs += [0.0]
        return torch.FloatTensor(sigs).to(self.sigmas.device, self.dtype)

class Euler_Karras(Euler):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_Exponential(Euler):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_Uniform(Euler):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_Karras(Euler_a):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_Exponential(Euler_a):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_Uniform(Euler_a):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_CFG_PP_Karras(Euler_a_CFG_PP):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_CFG_PP_Exponential(Euler_a_CFG_PP):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class Euler_a_CFG_PP_Uniform(Euler_a_CFG_PP):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2M_Karras(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2M_Exponential(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2M_Uniform(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2S_a_Karras(DPM_2S_a):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2S_a_Exponential(DPM_2S_a):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2S_a_Uniform(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_SDE_Karras(DPM_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_SDE_Exponential(DPM_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_SDE_Uniform(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)
class DPM_2M_SDE_Karras(DPM_2M_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2M_SDE_Exponential(DPM_2M_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_2M_SDE_Uniform(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)
class DPM_3M_SDE_Karras(DPM_3M_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_3M_SDE_Exponential(DPM_3M_SDE):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class DPM_3M_SDE_Uniform(DPM_2M):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class LCM_Karras(LCM):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerKarras().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class LCM_Exponential(LCM):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerExponential().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)

class LCM_Uniform(LCM):
    def __init__(self, model, eta=1, scheduler=None):
        scheduler = scheduler or SchedulerUniform().to(model.device, model.dtype)
        super().__init__(model, eta, scheduler)