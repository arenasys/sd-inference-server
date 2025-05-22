import torch

class GuidedDenoiser():
    def __init__(self, unet, device, conditioning_schedule, scale, cfg_rescale, prediction_type=None):
        self.unet = unet
        self.conditioning_schedule = conditioning_schedule
        self.conditioning = None
        self.additional_conditioning = None
        self.additional_kwargs = None
        self.scale = scale

        self.mask = None
        self.original = None

        self.device = device
        self.dtype = unet.dtype

        self.cfg_rescale = cfg_rescale
        self.override_prediction_type = prediction_type

        self.cfg_pp = False
        self.uncond_pred = None

        self.predictions = None

        self.inpainting_input = None

        self.get_conditioning()

    def get_prediction_type(self):
        prediction_type = self.unet.prediction_type

        if self.override_prediction_type:
            prediction_type = self.override_prediction_type

        if prediction_type == "unknown":
            self.unet.determine_type()
            prediction_type = self.unet.prediction_type
        
        return prediction_type

    def get_conditioning(self):
        self.compositions = self.conditioning_schedule.get_compositions(self.dtype, self.device)

    def set_unet(self, unet):
        self.unet = unet

    def set_mask(self, mask, original):
        self.mask = mask.to(self.dtype)
        self.original = original.to(self.dtype)

    def set_inpainting(self, masked, masks):
        self.inpainting_input = torch.cat([masks, masked], dim=1).to(self.device, self.dtype)

    def set_scale(self, scale):
        self.scale = scale
    
    def set_cfg_rescale(self, cfg_rescale):
        self.cfg_rescale = cfg_rescale
    
    def set_cfg_pp(self, cfg_pp):
        self.cfg_pp = cfg_pp

    def set_prediction_type(self, prediction_type):
        self.override_prediction_type = prediction_type

    def set_predictions(self, predictions):
        self.predictions = predictions

    def predict(self, latents, timestep, conditioning):
        timestep = torch.ceil_(timestep)
        inputs = self.get_additional_inputs(latents)
        return self.unet(inputs, timestep, encoder_hidden_states=conditioning,
                         added_cond_kwargs=self.additional_conditioning,
                         added_cross_kwargs=self.additional_kwargs).sample

    def predict_noise_epsilon(self, latents, timestep, conditioning, alpha):
        noise_pred = self.predict(latents, timestep, conditioning)
        return noise_pred

    def predict_noise_v(self, latents, timestep, conditioning, alpha):
        v_pred = self.predict(latents, timestep, conditioning)
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
        composed_uncond_pred = [] if self.cfg_pp else None
        i = 0
        for (masks, pos_weights), neg_weights in self.compositions:
            pos_len, neg_len = len(pos_weights), len(neg_weights)
            pos = pred[i:i+pos_len]
            neg = pred[i+pos_len:i+pos_len+neg_len]

            # Apply composition
            neg = (neg*neg_weights).sum(dim=0, keepdims=True) / torch.sum(neg_weights)
            pos = pos * masks + (neg * (1 - masks))

            scale = self.scale

            # CFG++
            if self.cfg_pp:
                composed_uncond_pred += [neg]

            # Apply CFG
            cfg = neg + ((pos - neg) * (pos_weights * scale)).sum(dim=0, keepdims=True)

            # Rescale CFG
            if self.cfg_rescale:
                std = (torch.std(pos)/torch.std(cfg))
                factor = self.cfg_rescale*std + (1-self.cfg_rescale)
                cfg = cfg * factor
            
            composed_pred += [cfg]
            i += pos_len + neg_len
        
        if self.cfg_pp:
            self.uncond_pred = torch.cat(composed_uncond_pred)
        
        return torch.cat(composed_pred)

    def predict_noise(self, latents, timestep, alpha):
        model_input = self.get_model_inputs(latents)
        conditioning = self.conditioning

        prediction_type = self.get_prediction_type()

        if prediction_type == "epsilon":
            noise_pred = self.predict_noise_epsilon(model_input, timestep, conditioning, alpha)
        elif prediction_type == "v":
            noise_pred = self.predict_noise_v(model_input, timestep, conditioning, alpha)

        composed_pred = self.compose_predictions(noise_pred)
        return composed_pred

    def predict_original_epsilon(self, latents, timestep, sigma, conditioning):
        c_in = 1 / (sigma ** 2 + 1) ** 0.5

        noise_pred = self.predict(latents * c_in, timestep, conditioning)
        original_pred = latents - sigma * noise_pred
        return original_pred

    def predict_original_v(self, latents, timestep, sigma, conditioning):
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_skip = 1 / (sigma ** 2 + 1)
        c_out = -sigma * 1 / (sigma ** 2 + 1) ** 0.5

        v_pred = self.predict(latents * c_in, timestep, conditioning)
        original_pred = v_pred * c_out + latents * c_skip
        return original_pred

    def mask_original(self, original_pred):
        if self.mask != None:
            original_pred = (self.original * self.mask) + (original_pred * (1 - self.mask))
        return original_pred

    def predict_original(self, latents, timestep, sigma):
        model_input = self.get_model_inputs(latents)
        conditioning = self.conditioning

        prediction_type = self.get_prediction_type()

        if prediction_type == "epsilon":
            original_pred = self.predict_original_epsilon(model_input, timestep, sigma, conditioning)
        elif prediction_type == "v":
            original_pred = self.predict_original_v(model_input, timestep, sigma, conditioning)
        else:
            raise RuntimeError(f"Unknown prediction type: {prediction_type}")

        composed_pred = self.compose_predictions(original_pred)
        masked_pred = self.mask_original(composed_pred)
        return masked_pred

    def set_step(self, step):
        self.conditioning = self.conditioning_schedule.get_conditioning_at_step(step, self.dtype, self.device)
        self.additional_conditioning = self.conditioning_schedule.get_additional_conditioning_at_step(step, self.dtype, self.device)
        self.additional_kwargs = self.conditioning_schedule.get_additional_attention_kwargs_at_step(step)
        self.unet.additional.set_strength(self.conditioning_schedule.get_networks_at_step(step))
        
    def reset(self):
        self.mask = None
        self.original = None
        self.conditioning = None
        self.get_conditioning()