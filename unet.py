import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import FeedForward

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

import einops

class SDUpsample2D(nn.Module):
    def __init__(self, channels, out_channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class SDDownsample2D(nn.Module):
    def __init__(self, channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=2, padding=padding)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states

class SDCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))
    
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
class SDBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()

        # 1. Self-Attn
        self.attn1 = SDCrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = SDCrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        if cross_attention_dim is not None:
            self.norm2 = (
                nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states
    ):
        norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=norm_hidden_states
        )

        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class SDTransformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 2. Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SDBasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states
    ):
        # 1. Input
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        return output


class SDCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        self.num_layers = num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SDTransformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    SDDownsample2D(
                        out_channels, out_channels=out_channels, padding=downsample_padding
                    )
                ]
            )
        else:
            self.downsamplers = None

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

        return hidden_states, (states_0, states_1, states_2)

class SDDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
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
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    SDDownsample2D(
                        out_channels, out_channels=out_channels, padding=downsample_padding
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb):
        hidden_states = self.resnets[0](hidden_states, temb)
        state_0 = hidden_states

        hidden_states = self.resnets[1](hidden_states, temb)
        state_1 = hidden_states

        return hidden_states, (state_0, state_1)

class SDUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
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
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([SDUpsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], temb):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class SDCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()

        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SDTransformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([SDUpsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        temb,
        encoder_hidden_states
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class SDUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                SDTransformer2DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = self.resnets[0](hidden_states, temb)

        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)

        return hidden_states

class SDUNET(nn.Module):
    def __init__(self, cross_attention_dim, attention_head_dim, use_linear_projection=False):
        super().__init__()

        self.time_proj = Timesteps(320, True, 0.0)
        self.time_embedding = TimestepEmbedding(320, 1280)

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        common = {'temb_channels': 1280, 'resnet_eps': 1e-05, 'resnet_act_fn': 'silu', 'resnet_groups': 32}
        cross = {'cross_attention_dim': cross_attention_dim, 'use_linear_projection': use_linear_projection}

        down0 = SDCrossAttnDownBlock2D(320, 320, num_layers=2, attn_num_head_channels=attention_head_dim[0], **cross, **common)
        down1 = SDCrossAttnDownBlock2D(320, 640, num_layers=2,  attn_num_head_channels=attention_head_dim[1], **cross, **common)
        down2 = SDCrossAttnDownBlock2D(640, 1280, num_layers=2,  attn_num_head_channels=attention_head_dim[2], **cross, **common)
        down3 = SDDownBlock2D(1280, 1280, num_layers=2,  add_downsample=False, **common)
        self.down_blocks.append(down0).append(down1).append(down2).append(down3)

        self.mid_block = SDUNetMidBlock2DCrossAttn(1280, num_layers=1, attn_num_head_channels=attention_head_dim[3], output_scale_factor=1, **cross, **common)
        up0 = SDUpBlock2D(1280, 1280, 1280, num_layers=3, **common)
        up1 = SDCrossAttnUpBlock2D(640, 1280, 1280, num_layers=3, attn_num_head_channels=attention_head_dim[2], **cross, **common)
        up2 = SDCrossAttnUpBlock2D(320, 640, 1280, num_layers=3, attn_num_head_channels=attention_head_dim[1], **cross, **common)
        up3 = SDCrossAttnUpBlock2D(320, 320, 640, num_layers=3, add_upsample=False, attn_num_head_channels=attention_head_dim[0], **cross, **common)
        self.up_blocks.append(up0).append(up1).append(up2).append(up3)

        self.conv_norm_out = nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        timesteps = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        sample_original = sample
        sample, down_sample_0 = self.down_blocks[0](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, down_sample_1 = self.down_blocks[1](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, down_sample_2 = self.down_blocks[2](hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, down_sample_3 = self.down_blocks[3](hidden_states=sample, temb=emb)

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        
        up_sample_0 = (down_sample_2[2], *down_sample_3)
        up_sample_1 = (down_sample_1[2], *down_sample_2[0:2])
        up_sample_2 = (down_sample_0[2], *down_sample_1[0:2])
        up_sample_3 =  (sample_original, *down_sample_0[0:2])

        sample = self.up_blocks[0](res_hidden_states_tuple=up_sample_0, hidden_states=sample, temb=emb)
        sample = self.up_blocks[1](res_hidden_states_tuple=up_sample_1, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample = self.up_blocks[2](res_hidden_states_tuple=up_sample_2, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample = self.up_blocks[3](res_hidden_states_tuple=up_sample_3, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample