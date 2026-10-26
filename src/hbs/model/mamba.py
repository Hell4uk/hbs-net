"""
Selective State Space Model (Mamba2-style) block in pure PyTorch.

Supports two execution modes:
- sequence mode: process a whole sequence in parallel (training);
- step mode: process one token at a time with a recurrent state (inference).

The parallel scan uses a chunked formulation of the linear recurrence
h_t = a_t * h_{t-1} + b_t: within each chunk the cumulative log-decay range
is bounded so all exponentials stay in float32 range, and the recurrent
state is carried between chunks sequentially.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbs.config import ModelConfig


class SelectiveSSM(nn.Module):
    """Diagonal selective SSM with input-dependent B, C and dt."""

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            padding=3,
            groups=d_model,
            bias=True,
        )
        self.x_proj = nn.Linear(d_model, 3 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, d_model, bias=True)

        # A is diagonal, stored as log of positive values; A = -exp(A_log).
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def A(self) -> torch.Tensor:
        return -torch.exp(self.A_log)  # (d_state,)

    def _discretize(
        self, B: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-order hold discretization.

        B: (..., d_state), dt: (..., d_model)
        Returns:
            A_bar: (..., d_model, d_state) decay per channel
            B_bar: (..., d_model, d_state) input per channel
        """
        dt = F.softplus(dt)
        A = self.A  # (d_state,)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(-2)
        return A_bar, B_bar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence mode. x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv.
        x_in = x_in.transpose(1, 2)  # (batch, d_model, seq)
        # With padding=3 and kernel=4, output[t] already covers the
        # window x[t-3..t], so the first `seq` outputs are causal.
        x_in = self.conv1d(x_in)[..., : x_in.shape[-1]]
        x_in = x_in.transpose(1, 2)  # (batch, seq, d_model)
        x_in = F.silu(x_in)

        B, C, dt = self.x_proj(x_in).chunk(3, dim=-1)
        dt = self.dt_proj(dt)  # (batch, seq, d_model)

        A_bar, B_bar = self._discretize(B, dt)
        C_exp = C.unsqueeze(-2)  # (batch, seq, d_model, d_state)

        h = self._parallel_scan(A_bar, B_bar * x_in.unsqueeze(-1))
        y = (h * C_exp).sum(dim=-1) + self.D * x_in
        y = y * F.silu(z)
        return self.out_proj(y)

    def _parallel_scan(
        self, A_bar: torch.Tensor, B_bar: torch.Tensor
    ) -> torch.Tensor:
        """Compute h_t = A_bar_t * h_{t-1} + B_bar_t over the sequence dim.

        A_bar, B_bar: (batch, seq, d_model, d_state) -> h: same shape.
        """
        # The (batch, seq, d_model, d_state) intermediates are large;
        # process small batch slices to bound peak memory.
        batch = A_bar.shape[0]
        if batch > 8:
            return torch.cat(
                [
                    self._scan_single(A_bar[i : i + 8], B_bar[i : i + 8])
                    for i in range(0, batch, 8)
                ],
                dim=0,
            )
        return self._scan_single(A_bar, B_bar)

    def _scan_single(
        self, A_bar: torch.Tensor, B_bar: torch.Tensor
    ) -> torch.Tensor:
        """Scan for a batch of up to 8 items: (b, s, d, n)."""
        batch, seq, d_model, d_state = A_bar.shape
        log_A = torch.log(A_bar.clamp_min(1e-8))  # (b, s, d, n), <= 0

        max_decay = (-log_A).max().item()
        chunk = max(1, min(seq, int(60.0 / max(max_decay, 1e-6))))

        h = torch.zeros(
            batch, d_model, d_state, dtype=A_bar.dtype, device=A_bar.device
        )
        outputs: list[torch.Tensor] = []
        for start in range(0, seq, chunk):
            end = min(start + chunk, seq)
            log_c = log_A[:, start:end]
            cum_c = torch.cumsum(log_c, dim=1)  # (b, c, d, n), <= 0
            neg_c = -cum_c
            # Reference: value at the chunk's last position (per d, n).
            ref = neg_c[:, -1:].amax(dim=1, keepdim=True)  # (b, 1, d, n)
            scaled = torch.exp(neg_c - ref) * B_bar[:, start:end]
            prefix = torch.cumsum(scaled, dim=1)
            local = torch.exp(cum_c + ref) * prefix
            # Carry previous state: h_t += exp(cum_c[t]) * h_{start-1}
            decayed = torch.exp(cum_c) * h.unsqueeze(1)
            out_c = local + decayed
            outputs.append(out_c)
            h = out_c[:, -1]
        return torch.cat(outputs, dim=1)

    def step(
        self, x_t: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step mode. x_t: (batch, 1, d_model).

        state: (batch, d_model, d_state + conv_window) or None.
        Returns (y_t, new_state); y_t: (batch, 1, d_model).
        """
        x_t = x_t.squeeze(1)
        batch = x_t.shape[0]
        conv_window = self.conv1d.kernel_size[0] - 1
        if state is None:
            state = torch.zeros(
                batch,
                self.d_model,
                self.d_state + conv_window,
                dtype=x_t.dtype,
                device=x_t.device,
            )
        ssm_state = state[..., : self.d_state]
        conv_state = state[..., self.d_state :]

        xz = self.in_proj(x_t)
        x_in, z = xz.chunk(2, dim=-1)

        # Causal conv in step mode: slide the stored window.
        window = torch.cat([conv_state, x_in.unsqueeze(-1)], dim=-1)
        # The window has length kernel_size; its last output position
        # covers the full window, matching sequence-mode output[t].
        x_conv = self.conv1d(window)[..., conv_window]
        new_conv_state = window[..., 1:]
        x_in = F.silu(x_conv)

        B, C, dt = self.x_proj(x_in).chunk(3, dim=-1)
        dt = self.dt_proj(dt)

        A_bar, B_bar = self._discretize(B, dt)
        # x_t has seq dim 1; drop it so the state stays 3-D.
        A_bar = A_bar.squeeze(1)
        B_bar = B_bar.squeeze(1)
        new_ssm_state = A_bar * ssm_state + B_bar * x_in.unsqueeze(-1)
        new_state = torch.cat([new_ssm_state, new_conv_state], dim=-1)
        y = (new_ssm_state * C.unsqueeze(-2)).sum(dim=-1) + self.D * x_in
        y = y * F.silu(z)
        return self.out_proj(y).unsqueeze(1), new_state


class MambaBlock(nn.Module):
    """Pre-norm residual block wrapping SelectiveSSM."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.ssm = SelectiveSSM(config.d_model, config.d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))

    def step(
        self, x_t: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, new_state = self.ssm.step(self.norm(x_t), state)
        return x_t + h, new_state
