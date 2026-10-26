"""
Causal self-attention block with KV-cache support for generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbs.config import ModelConfig


class AttentionBlock(nn.Module):
    """Pre-norm residual block: multi-head causal self-attention + FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attend(self.norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def _attend(self, x: torch.Tensor, kv_cache: list[torch.Tensor] | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if kv_cache is not None:
            if kv_cache:
                k = torch.cat([kv_cache[0], k], dim=1)
                v = torch.cat([kv_cache[1], v], dim=1)
            kv_cache.clear()
            kv_cache.extend([k, v])

        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(kv_cache is None and seq > 1)
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.out_proj(out)

    def step(
        self, x_t: torch.Tensor, kv_cache: list[torch.Tensor] | None
    ) -> torch.Tensor:
        """Single-token step for generation. x_t: (batch, 1, d_model)."""
        x_t = x_t + self._attend(self.norm(x_t), kv_cache=kv_cache)
        x_t = x_t + self.ffn(self.ffn_norm(x_t))
        return x_t
