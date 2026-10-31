"""Decoder-only transformer, adapted from nanoGPT (Karpathy, 2022; MIT license,
https://github.com/karpathy/nanoGPT, model.py at 3adf61e).

Changes from nanoGPT: separate q/k/v projections (so Muon orthogonalizes each and the Hessian
anatomy can split QK from OV), optional weight tying (off by default), no dropout, logits for every
position, and `forward_embeds` for derivatives with respect to the input embeddings.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    bias: bool = False
    tie_weights: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = (proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for proj in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init of the residual projections, as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                torch.nn.init.zeros_(module.bias)

    def embed(self, idx):
        return self.transformer.wte(idx)

    def forward_embeds(self, tok_emb):
        T = tok_emb.size(1)
        assert T <= self.config.block_size
        pos = torch.arange(T, device=tok_emb.device)
        x = tok_emb + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

    def forward(self, idx):
        return self.forward_embeds(self.embed(idx))

    def loss(self, idx, targets):
        logits = self(idx)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
