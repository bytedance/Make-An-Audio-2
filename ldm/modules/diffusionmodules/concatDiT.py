# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) Make-An-Audio-2 Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import math
import collections.abc
from itertools import repeat
from einops import rearrange
from ldm.modules.new_attention import CrossAttention,Conv1dFeedForward,checkpoint,Normalize,zero_module,PositionEmbedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(repeat(x, 2))


################################################################
#               Embedding Layers for Timesteps                 #
################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Conv1DFinalLayer(nn.Module):
    """
    The final layer of CrossAttnDiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.GroupNorm(16,hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, out_channels,kernel_size=1)

    def forward(self, x): # x:(B,C,T)
        x = self.norm_final(x)
        x = self.conv1d(x)
        return x

class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, context_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_size, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size)
        )

    def forward(self,x):
        return self.mlp(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True): # 1 self 1 cross or 2 self
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention,if context is none
        self.ff = Conv1dFeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # use as cross attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):# x shape:(B,T,C)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x)) + x

        x = self.ff(self.norm3(x).permute(0,2,1)).permute(0,2,1) + x
        return x

class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head 
        self.norm = Normalize(in_channels)
        
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))# initialize with zero

    def forward(self, x):# x shape (b,c,t)
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)# group norm
        x = self.proj_in(x)# no shape change
        x = rearrange(x,'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x)# context shape [b,seq_len=77,context_dim]
        x = rearrange(x,'b t c -> b c t')
        
        x = self.proj_out(x)
        x = x + x_in
        return x

class ConcatDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels # vae dim
        self.out_channels =  in_channels 
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionEmbedder(hidden_size,context_dim)
        self.proj_in = nn.Conv1d(in_channels,hidden_size,kernel_size=kernel_size,padding=kernel_size//2)

        self.pos_emb = PositionEmbedding(num_embeddings=max_len,embedding_dim = hidden_size)
        self.blocks = nn.ModuleList([
            TemporalTransformer(hidden_size,num_heads,d_head=hidden_size//num_heads,depth=1,context_dim=context_dim) for _ in range(depth)
        ])

        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear): # 
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        t = self.t_embedder(t).unsqueeze(1)  # (N,1,hidden_size)
        c = self.c_embedder(context)  # (N,c_len,hidden_size)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x,'b c t -> b t c')
        x = torch.concat([t,c,x],dim=1)
        x = self.pos_emb(x)
        x = rearrange(x,'b t c -> b c t')
        for block in self.blocks:
            x = block(x)                      # (N, D, extra_len+T)
        x = x[...,extra_len:] # (N,D,T)
        x = self.final_layer(x)                # (N, out_channels,T)
        return x

class ConcatDiT2MLP(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels # vae dim
        self.out_channels =  in_channels 
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c1_embedder = ConditionEmbedder(hidden_size,context_dim)
        self.c2_embedder = ConditionEmbedder(hidden_size,context_dim)
        self.proj_in = nn.Conv1d(in_channels,hidden_size,kernel_size=kernel_size,padding=kernel_size//2)

        self.pos_emb = PositionEmbedding(num_embeddings=max_len,embedding_dim = hidden_size)
        self.blocks = nn.ModuleList([
            TemporalTransformer(hidden_size,num_heads,d_head=hidden_size//num_heads,depth=1,context_dim=context_dim) for _ in range(depth)
        ])

        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear): # 
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        t = self.t_embedder(t).unsqueeze(1)  # (N,1,hidden_size)
        c1,c2 = context.chunk(2,dim=1)
        c1 = self.c1_embedder(c1)  # (N,c_len,hidden_size)
        c2 = self.c2_embedder(c2)  # (N,c_len,hidden_size)
        c = torch.cat((c1,c2),dim=1)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x,'b c t -> b t c')
        x = torch.concat([t,c,x],dim=1)
        x = self.pos_emb(x)
        x = rearrange(x,'b t c -> b c t')
        for block in self.blocks:
            x = block(x)                      # (N, D, extra_len+T)
        x = x[...,extra_len:] # (N,D,T)
        x = self.final_layer(x)                # (N, out_channels,T)
        return x

