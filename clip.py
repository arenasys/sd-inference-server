
import torch
from transformers.models.clip.modeling_clip import CLIPTextTransformer, _create_4d_causal_attention_mask, BaseModelOutputWithPooling

class CustomCLIP(torch.nn.Module):
    def __init__(self, config, text_projection=False):
        super().__init__()
        self.text_model = CustomCLIPTextTransformer(config)
        if text_projection:
            self.text_projection = torch.nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)

    def forward(self, input_ids, clip_skip=1):
        output = self.text_model(input_ids).hidden_states[-clip_skip]
        cond = self.text_model.final_layer_norm(output)
        emb = None
        return cond, emb

class CustomSDXLCLIP(torch.nn.Module):
    def __init__(self, open_clip_config, ldm_clip_config):
        super().__init__()
        self.open_clip = CustomCLIP(open_clip_config, text_projection=True)
        self.ldm_clip = CustomCLIP(ldm_clip_config)

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)
    
    def forward(self, input_ids, clip_skip=1):
        clip_skip = 2

        open_clip_input_ids = input_ids
        ldm_clip_input_ids = [49407 if type(i) == int and i == 0 else i for i in input_ids]

        open_clip_input_ids = [i[:1280] if type(i) == torch.Tensor else i for i in open_clip_input_ids]
        ldm_clip_input_ids = [i[1280:] if type(i) == torch.Tensor else i for i in ldm_clip_input_ids]

        open_clip_outputs = self.open_clip.text_model(open_clip_input_ids)
        ldm_clip_outputs = self.ldm_clip.text_model(ldm_clip_input_ids)

        open_clip_cond = open_clip_outputs.hidden_states[-clip_skip]
        ldm_clip_cond = ldm_clip_outputs.hidden_states[-clip_skip]
        cond = torch.cat([ldm_clip_cond, open_clip_cond], dim=2)
        
        emb = self.open_clip.text_projection(open_clip_outputs.pooler_output)[0]

        return cond, emb

class CustomCLIPTextTransformer(CLIPTextTransformer):
    # needed to nicely handle mixed tokens and embeddings
    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
        output_attentions = None, output_hidden_states = True, return_dict = True):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # tensors are already embedded so bypass the token_embedding
        tensors = [t if type(t) == torch.Tensor else None for t in input_ids]
        input_ids = [t if type(t) != torch.Tensor else 0 for t in input_ids]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        inputs_embeds = [e for e in self.embeddings.token_embedding(input_ids)[0]]
        for i in range(len(inputs_embeds)):
            if tensors[i] != None:
                inputs_embeds[i] = tensors[i].to(inputs_embeds[i].dtype)
        inputs_embeds = torch.stack(inputs_embeds).unsqueeze(0)

        position_ids = self.embeddings.position_ids[:, :inputs_embeds.shape[1]]
        position_embeddings = self.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.size(), hidden_states.dtype, device=hidden_states.device
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=input_ids.device), input_ids.to(torch.int).argmax(dim=-1)]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)