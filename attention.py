import torch
import einops
import math
import diffusers

HAVE_XFORMERS = False
try:
    import xformers
    HAVE_XFORMERS = True
except Exception:
    pass

ORIGINAL_FORWARD = None
try:
    ORIGINAL_FORWARD = diffusers.models.attention_processor.Attention.forward
except:
    pass
try:
    ORIGINAL_FORWARD = diffusers.models.attention.CrossAttention.forward
except:
    pass

def set_attention(forward):
    try:
        diffusers.models.attention_processor.Attention.forward = forward
        return
    except:
        pass
    try:
        diffusers.models.attention.CrossAttention.forward = forward
        return
    except:
        pass
    raise ValueError("failed to find attention forward")

def get_available():
    available = [
        use_optimized_attention,
        use_split_attention_v1,
        use_split_attention,
        use_flash_attention,
        use_diffusers_attention
    ]
    if HAVE_XFORMERS:
        available += [use_xformers_attention]
    return available

def use_optimized_attention(device):
    if "cuda" in str(device):
        try:
            use_xformers_attention(device)
        except Exception:
            use_split_attention(device)
    else:
       use_split_attention_v1(device)

def use_diffusers_attention(device):
    set_attention(ORIGINAL_FORWARD)
    
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def use_split_attention(device):
    if not "cuda" in str(device):
       raise RuntimeError("Split v2 attention does not support CPU generation")

    def get_available_vram(device):
        stats = torch.cuda.memory_stats(device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total

    def split_cross_attention_forward(self, x, encoder_hidden_states=None, attention_mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(encoder_hidden_states, x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        k_in = self.to_k(context)
        v_in = self.to_v(context)
        del context, x

        dtype = q_in.dtype
        device = q_in.device


        k_in = k_in * self.scale
    
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in
    
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    
        mem_free_total = get_available_vram(device)
    
        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1
    
        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
    
        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')
    
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end], k)
    
            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1
    
            r1[:, i:end] = torch.einsum('b i j, b j d -> b i d', s2, v)
            del s2
    
        del q, k, v

        r1 = r1.to(dtype)

        r2 = einops.rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        out = self.to_out[0](r2)
        out = self.to_out[1](out)

        return out

    set_attention(split_cross_attention_forward)

def use_split_attention_v1(device):
    def split_cross_attention_forward_v1(self, x, encoder_hidden_states=None, attention_mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(encoder_hidden_states, x)

        k_in = self.to_k(context)
        v_in = self.to_v(context)
        del context, x
        
        q0, q1, q2 = q_in.shape
        q = q_in.reshape(q0, q1, h, q2//h)
        q = q.permute(0, 2, 1, 3)
        q = q.reshape(q0*h, q1, q2//h)
        del q_in, q0, q1, q2

        k0, k1, k2 = k_in.shape
        k = k_in.reshape(k0, k1, h, k2//h)
        k = k.permute(0, 2, 1, 3)
        k = k.reshape(k0*h, k1, k2//h)
        del k_in, k0, k1, k2

        v0, v1, v2 = v_in.shape
        v = v_in.reshape(v0, v1, h, v2//h)
        v = v.permute(0, 2, 1, 3)
        v = v.reshape(v0*h, v1, v2//h)
        del v_in, v0, v1, v2

        dtype = q.dtype

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = torch.einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale
            s2 = s1.softmax(dim=-1)
            r1[i:end] = torch.einsum('b i j, b j d -> b i d', s2, v[i:end])
        
        del q, k, v

        r1 = r1.to(dtype)

        z0, z1, z2 = r1.shape
        r2 = r1.reshape(z0//h, h, z1, z2)
        r2 = r2.permute(0, 2, 1, 3)
        r2 = r2.reshape(z0//h, z1, z2*h)
        del r1
        
        out = self.to_out[0](r2)
        out = self.to_out[1](out)

        return out

    set_attention(split_cross_attention_forward_v1)

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
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(q_start_index - k_start_index + 1)
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

def use_flash_attention(device):
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, x, encoder_hidden_states=None, attention_mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = default(encoder_hidden_states, x)
        context = context.to(x.dtype)

        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = flash_func.apply(q, k, v, attention_mask, False, q_bucket_size, k_bucket_size)

        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    set_attention(forward_flash_attn)

def use_xformers_attention(device):
    if not "cuda" in str(device):
        raise RuntimeError("XFormers attention does not support CPU generation")
    if not HAVE_XFORMERS:
        raise RuntimeError("XFormers not installed")

    def xformers_attention_forward(self, x, encoder_hidden_states=None, attention_mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = default(encoder_hidden_states, x)

        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        dtype = q.dtype

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

        out = out.to(dtype)

        out = einops.rearrange(out, 'b n h d -> b n (h d)', h=h)

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
    
    set_attention(xformers_attention_forward)