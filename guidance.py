import torch

class GuidedDenoiser():
    def __init__(self, unet, conditioning_schedule, scale):
        self.unet = unet
        self.conditioning_schedule = conditioning_schedule
        self.conditioning = None
        self.scale = scale

        self.mask = None
        self.original = None

        self.device = unet.device
        self.dtype = unet.dtype

        self.predictions = None

        self.inpainting_input = None

        self.get_compositions()

    def get_compositions(self):
        self.compositions = self.conditioning_schedule.get_compositions(self.dtype, self.device)

    def set_mask(self, mask, original):
        if mask != None:
            self.mask = mask.to(self.dtype)
            self.original = original.to(self.dtype) * 0.18215

    def set_inpainting(self, masked, masks):
        self.inpainting_input = torch.cat([masks, masked], dim=1).to(self.device, self.dtype)

    def set_predictions(self, predictions):
        self.predictions = predictions

    def predict_noise_epsilon(self, latents, timestep, conditioning, alpha):
        inputs = self.get_additional_inputs(latents)
        noise_pred = self.unet(inputs, timestep, encoder_hidden_states=conditioning).sample
        return noise_pred

    def predict_noise_v(self, latents, timestep, conditioning, alpha):
        inputs = self.get_additional_inputs(latents)
        v_pred = self.unet(inputs, timestep, encoder_hidden_states=conditioning).sample
        return alpha.sqrt() * v_pred + (1-alpha).sqrt() * latents

    def mask_noise(self, latents, alpha, noise):
        if self.mask != None:
            noised_original = alpha.sqrt() * self.original + (1-alpha).sqrt() * noise()
            latents = (noised_original * self.mask) + (latents * (1 - self.mask))
        return latents
    
    def get_model_inputs(self, latents):
        model_input = []
        for i, ((m, p), n) in enumerate(self.compositions):
            model_input += [latents[i:i+1]]*(len(p) + len(n))
        model_input = torch.cat(model_input)

        return model_input
    
    def get_additional_inputs(self, latents):
        if self.inpainting_input != None:
            inpainting_inputs = torch.cat([self.inpainting_input]*latents.shape[0])
            latents = torch.cat([latents, inpainting_inputs], dim=1)
        return latents

    def compose_predictions(self, pred):
        composed_pred = []
        i = 0
        for (m, p), n in self.compositions:
            pl, nl = len(p), len(n)
            neg = (pred[i+pl:i+pl+nl]*n).sum(dim=0, keepdims=True) / torch.sum(n)
            pred[i:i+pl] = pred[i:i+pl] * m + (neg * (1 - m))
            composed_pred += [neg + ((pred[i:i+pl] - neg) * (p * self.scale)).sum(dim=0, keepdims=True)]
            i += pl + nl
        return torch.cat(composed_pred)

    def predict_noise(self, latents, timestep, alpha):
        model_input = self.get_model_inputs(latents)
        conditioning = self.conditioning

        if self.unet.prediction_type == "epsilon":
            noise_pred = self.predict_noise_epsilon(model_input, timestep, conditioning, alpha)
        elif self.unet.prediction_type == "v":
            noise_pred = self.predict_noise_v(model_input, timestep, conditioning, alpha)

        composed_pred = self.compose_predictions(noise_pred)
        return composed_pred

    def predict_original_epsilon(self, latents, timestep, sigma, conditioning):
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        inputs = self.get_additional_inputs(latents * c_in)
        noise_pred = self.unet(inputs, timestep, encoder_hidden_states=conditioning).sample
        original_pred = latents - sigma * noise_pred
        return original_pred

    def predict_original_v(self, latents, timestep, sigma, conditioning):
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_skip = 1 / (sigma ** 2 + 1)
        c_out = -sigma * 1 / (sigma ** 2 + 1) ** 0.5

        inputs = self.get_additional_inputs(latents * c_in)
        v_pred = self.unet(inputs, timestep, encoder_hidden_states=conditioning).sample
        original_pred = v_pred * c_out + latents * c_skip
        return original_pred

    def mask_original(self, original_pred):
        if self.mask != None:
            original_pred = (self.original * self.mask) + (original_pred * (1 - self.mask))
        return original_pred

    def predict_original(self, latents, timestep, sigma):
        model_input = self.get_model_inputs(latents)
        conditioning = self.conditioning

        if self.unet.prediction_type == "epsilon":
            original_pred = self.predict_original_epsilon(model_input, timestep, sigma, conditioning)
        elif self.unet.prediction_type == "v":
            original_pred = self.predict_original_v(model_input, timestep, sigma, conditioning)

        composed_pred = self.compose_predictions(original_pred)
        masked_pred = self.mask_original(composed_pred)
        return masked_pred

    def set_step(self, step):
        self.conditioning = self.conditioning_schedule.get_conditioning_at_step(step).to(self.dtype)
        nets = self.conditioning_schedule.get_networks_at_step(step)
        self.unet.additional.set_strength(nets)
        
    def reset(self):
        self.mask = None
        self.original = None
        self.conditioning = None
        self.get_compositions()