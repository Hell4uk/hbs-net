"""
HBSNet: Hybrid Belief-State Network.

Embedding -> SSM layers (Mamba) with Dialectical Gate on the last SSM layer
-> Attention layers -> tied LM head. Returns logits, contradiction logits
and state energy for the composite loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hbs.config import ModelConfig
from hbs.model.attention import AttentionBlock
from hbs.model.gate import DialecticalGate
from hbs.model.mamba import MambaBlock


@dataclass
class HBSOutput:
    logits: torch.Tensor            # (batch, seq, vocab)
    contradiction_logits: torch.Tensor | None  # (batch, seq) raw scores
    energy: torch.Tensor | None     # scalar mean state energy


class HBSNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.ssm_layers = nn.ModuleList(
            [MambaBlock(config) for _ in range(config.n_ssm_layers)]
        )
        self.gate = DialecticalGate(config)
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(config) for _ in range(config.n_attention_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tied

    def forward(
        self,
        tokens: torch.Tensor,
        contradiction_labels: torch.Tensor | None = None,
    ) -> HBSOutput:
        """tokens: (batch, seq) -> HBSOutput."""
        x = self.embedding(tokens)

        h_prev = x
        for layer in self.ssm_layers:
            h = layer(h_prev)
            h_prev = h

        # Dialectical gate on the final SSM state vs input embeddings.
        c, alpha = self.gate(h_prev, x)
        # Modulate the state: high contradiction -> stronger update signal.
        x_out = h_prev * (1.0 - alpha) + x * alpha

        for layer in self.attention_layers:
            x_out = layer(x_out)

        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)

        energy = (h_prev.detach().float() ** 2).mean()

        return HBSOutput(
            logits=logits,
            contradiction_logits=c.squeeze(-1),
            energy=energy,
        )

    @torch.no_grad()
    def generate_step(
        self,
        token: torch.Tensor,
        state: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Single generation step. token: (batch, 1) -> (logits, state).

        state: list of per-layer SSM states (or None) plus attention
        KV-caches; the exact layout is managed by the Generator.
        """
        x = self.embedding(token)
        h_prev = x
        new_state: list[torch.Tensor | None] = []
        idx = 0
        for layer in self.ssm_layers:
            h, s = layer.step(x, state[idx] if state else None)
            new_state.append(s)
            h_prev = h
            idx += 1

        # h_prev: (batch, 1, d) after residual; x: (batch, 1, d).
        c, alpha = self.gate(h_prev, x)
        x_out = h_prev * (1.0 - alpha) + x * alpha

        for layer in self.attention_layers:
            kv = state[idx] if state and idx < len(state) else None
            if kv is None:
                kv = []
            x_out = layer.step(x_out, kv_cache=kv)
            new_state.append(kv)
            idx += 1

        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)
        return logits, new_state
