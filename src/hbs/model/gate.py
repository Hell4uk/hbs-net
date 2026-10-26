"""
Dialectical Gate: contradiction index computation and state-update gating.

Implements the mechanism from the design document:
- c_i = MLP([h_{i-1}; x_i; h*x_i; h - x_i]) -> [0, 1]
- alpha_i = g_i * (beta + (1 - beta) * c_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hbs.config import ModelConfig


class DialecticalGate(nn.Module):
    """Computes a per-token contradiction index and update gate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.beta = config.contradiction_beta
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(
        self, h_prev: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h_prev, x: (batch, seq, d) -> (c, alpha) each (batch, seq, 1).

        In step mode h_prev may have seq=1 while x has seq=1 too; shapes
        must simply match. Mismatched seq lengths are broadcast over the
        last SSM state by the caller.
        """
        features = torch.cat(
            [h_prev, x, h_prev * x, h_prev - x], dim=-1
        )
        c = torch.sigmoid(self.mlp(features))
        # g: simple learned gate from x
        alpha = self.beta + (1.0 - self.beta) * c
        return c, alpha
