import torch
import einops
import math
import diffusers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def use_split_attention():
    def split_cross_attention_forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        k_in = self.to_k(context)
        v_in = self.to_v(context)
        del context, x
        
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = torch.einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale

            s2 = s1.softmax(dim=-1)
            del s1

            r1[i:end] = torch.einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        del q, k, v

        r2 = einops.rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        out = self.to_out[0](r2)
        out = self.to_out[1](out)

        return out

    diffusers.models.attention.CrossAttention.forward = split_cross_attention_forward

EPSILON = 1e-6

class FlashAttentionFunction(torch.autograd.Function):
  @ staticmethod
  @ torch.no_grad()
  def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
    """ Algorithm 2 in the paper """

    device = q.device
    dtype = q.dtype
    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    o = torch.zeros_like(q)
    all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
    all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

    scale = (q.shape[-1] ** -0.5)

    if not exists(mask):
      mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
    else:
      mask = einops.rearrange(mask, 'b n -> b 1 1 n')
      mask = mask.split(q_bucket_size, dim=-1)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        mask,
        all_row_sums.split(q_bucket_size, dim=-2),
        all_row_maxes.split(q_bucket_size, dim=-2),
    )

    for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if exists(row_mask):
          attn_weights.masked_fill_(~row_mask, max_neg_value)

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
        attn_weights -= block_row_maxes
        exp_weights = torch.exp(attn_weights)

        if exists(row_mask):
          exp_weights.masked_fill_(~row_mask, 0.)

        block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

        new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

        exp_values = torch.einsum('... i j, ... j d -> ... i d', exp_weights, vc)

        exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
        exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

        new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

        oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

        row_maxes.copy_(new_row_maxes)
        row_sums.copy_(new_row_sums)

    ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
    ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

    return o

def use_flash_attention():
  flash_func = FlashAttentionFunction

  def forward_flash_attn(self, x, context=None, mask=None):
    q_bucket_size = 512
    k_bucket_size = 1024

    h = self.heads
    q = self.to_q(x)

    context = context if context is not None else x
    context = context.to(x.dtype)

    k = self.to_k(context)
    v = self.to_v(context)
    del context, x

    q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

    out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    out = einops.rearrange(out, 'b h n d -> b n (h d)')

    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_flash_attn