import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Identity(nn.Module):
    def forward(self, x):
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample  

class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)

        emb = torch.exp(exponent / half_dim)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        return F.gelu(gate)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList([])
        self.net.append(GEGLU(dim, inner_dim))
        self.net.append(Identity())
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels=512,
        groups=32,
        eps=1e-5,
    ):
        super().__init__()

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = Identity()
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states)
        return output_tensor

class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states

class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=2, padding=padding)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        heads: int = 8,
        dim_head: int = 64
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(Identity())
    
    def forward(self, x, encoder_hidden_states):
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
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=dim,
        )

        self.ff = FeedForward(dim)

        self.attn2 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                cross_attention_dim=cross_attention_dim           
            )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states
    ):
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=norm_hidden_states)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        cross_attention_dim: int,
        num_layers: int = 1,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        self.out_channels = in_channels if out_channels is None else out_channels
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states
    ):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual

        return output


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attn_num_head_channels,
                    attention_head_dim=out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels=out_channels)])

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, temb, encoder_hidden_states
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        states_0 = hidden_states

        hidden_states = self.resnets[1](hidden_states, temb)
        hidden_states = self.attentions[1](hidden_states, encoder_hidden_states)
        states_1 = hidden_states
        
        hidden_states = self.downsamplers[0](hidden_states)
        states_2 = hidden_states

        return hidden_states, states_0, states_1, states_2

class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb):
        hidden_states = self.resnets[0](hidden_states, temb)
        state_0 = hidden_states

        hidden_states = self.resnets[1](hidden_states, temb)
        state_1 = hidden_states

        return hidden_states, state_0, state_1

class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])

    def forward(self, hidden_states, s0, s1, s2, temb):
        hidden_states = torch.cat([hidden_states, s0], dim=1)
        hidden_states = self.resnets[0](hidden_states, temb)

        hidden_states = torch.cat([hidden_states, s1], dim=1)
        hidden_states = self.resnets[1](hidden_states, temb)

        hidden_states = torch.cat([hidden_states, s2], dim=1)
        hidden_states = self.resnets[2](hidden_states, temb)

        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 1,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        add_upsample=True,
    ):
        super().__init__()

        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attn_num_head_channels,
                    attention_head_dim=out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = nn.ModuleList()

    def forward(self, hidden_states, s0, s1, s2, temb, encoder_hidden_states):

        hidden_states = self.resnets[0](torch.cat([hidden_states, s0], dim=1), temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = self.resnets[1](torch.cat([hidden_states, s1], dim=1), temb)
        hidden_states = self.attentions[1](hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = self.resnets[2](torch.cat([hidden_states, s2], dim=1), temb)
        hidden_states = self.attentions[2](hidden_states, encoder_hidden_states=encoder_hidden_states)

        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

        return hidden_states

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
    ):
        super().__init__()

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attn_num_head_channels,
                    attention_head_dim=in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = self.resnets[0](hidden_states, temb)

        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)

        return hidden_states

class UNET(nn.Module):
    def __init__(self, cross_attention_dim, attention_head_dim):
        super().__init__()

        self.time_proj = Timesteps(320)
        self.time_embedding = TimestepEmbedding(320, 1280)

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=(1, 1))

        down0 = CrossAttnDownBlock2D(320, 320, num_layers=2, attn_num_head_channels=attention_head_dim[0], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        down1 = CrossAttnDownBlock2D(320, 640, num_layers=2,  attn_num_head_channels=attention_head_dim[1], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        down2 = CrossAttnDownBlock2D(640, 1280, num_layers=2,  attn_num_head_channels=attention_head_dim[2], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        down3 = DownBlock2D(1280, 1280, num_layers=2, temb_channels=1280)
        self.down_blocks = nn.ModuleList([down0, down1, down2, down3])

        self.mid_block = UNetMidBlock2DCrossAttn(1280, num_layers=1, attn_num_head_channels=attention_head_dim[3], cross_attention_dim=cross_attention_dim, temb_channels=1280)

        up0 = UpBlock2D(1280, 1280, 1280, num_layers=3, temb_channels=1280)
        up1 = CrossAttnUpBlock2D(640, 1280, 1280, num_layers=3, attn_num_head_channels=attention_head_dim[2], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        up2 = CrossAttnUpBlock2D(320, 640, 1280, num_layers=3, attn_num_head_channels=attention_head_dim[1], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        up3 = CrossAttnUpBlock2D(320, 320, 640, num_layers=3, add_upsample=False, attn_num_head_channels=attention_head_dim[0], cross_attention_dim=cross_attention_dim, temb_channels=1280)
        self.up_blocks = nn.ModuleList([up0, up1, up2, up3])

        self.conv_norm_out = nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        timesteps = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        s0 = sample
        sample, s1_1, s1_2, s1_3 = self.down_blocks[0](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, s2_1, s2_2, s2_3 = self.down_blocks[1](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, s3_1, s3_2, s3_3 = self.down_blocks[2](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, s4_1, s4_2 = self.down_blocks[3](hidden_states=sample, temb=emb)

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        
        sample = self.up_blocks[0](s0=s4_2, s1=s4_1, s2=s3_3, hidden_states=sample, temb=emb)
        del s4_2, s4_1, s3_3
        sample = self.up_blocks[1](s0=s3_2, s1=s3_1, s2=s2_3, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        del s3_2, s3_1, s2_3
        sample = self.up_blocks[2](s0=s2_2, s1=s2_1, s2=s1_3, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        del s2_2, s2_1, s1_3
        sample = self.up_blocks[3](s0=s1_2, s1=s1_1, s2=s0, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        del s1_2, s1_1, s0

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def load_state_dict(self, state_dict, strict):
        # replace the linear projections in v2
        for k in state_dict:
            if k.endswith(".weight") and "proj_" in k:
                if len(state_dict[k].shape) == 2:
                    state_dict[k] = state_dict[k].unsqueeze(2).unsqueeze(2)

        return super().load_state_dict(state_dict, strict)