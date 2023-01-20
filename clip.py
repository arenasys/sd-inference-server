
import torch
from transformers.models.clip.modeling_clip import CLIPTextTransformer, _expand_mask, BaseModelOutputWithPooling

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

        bsz, seq_len = hidden_states.shape[:2]
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(hidden_states.device)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
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