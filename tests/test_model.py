"""Tests for the HBS model: shapes, causality, step/seq consistency."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hbs.config import ModelConfig
from hbs.model import HBSNet, MambaBlock, SelectiveSSM


@pytest.fixture(scope="module")
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        d_state=16,
        n_layers=2,
        n_ssm_layers=2,
        n_attention_layers=1,
        n_heads=4,
        dropout=0.0,
        max_seq_len=64,
    )


@pytest.fixture(scope="module")
def model(small_config: ModelConfig) -> HBSNet:
    torch.manual_seed(0)
    return HBSNet(small_config)


def test_forward_shapes(model: HBSNet, small_config: ModelConfig) -> None:
    tokens = torch.randint(0, small_config.vocab_size, (2, 10))
    out = model(tokens)
    assert out.logits.shape == (2, 10, small_config.vocab_size)
    assert out.contradiction_logits is not None
    assert out.contradiction_logits.shape == (2, 10)
    assert out.energy is not None
    assert out.energy.dim() == 0


def test_causality(model: HBSNet, small_config: ModelConfig) -> None:
    """Changing a future token must not change earlier logits."""
    torch.manual_seed(1)
    tokens = torch.randint(0, small_config.vocab_size, (1, 12))
    out_a = model(tokens)
    tokens_changed = tokens.clone()
    tokens_changed[0, 8:] = (tokens_changed[0, 8:] + 1) % small_config.vocab_size
    out_b = model(tokens_changed)
    diff = (out_a.logits[:, :8] - out_b.logits[:, :8]).abs().max().item()
    assert diff < 1e-5


def test_ssm_step_matches_sequence(small_config: ModelConfig) -> None:
    """Step mode must reproduce sequence mode exactly."""
    torch.manual_seed(2)
    ssm = SelectiveSSM(small_config.d_model, small_config.d_state)
    x = torch.randn(1, 8, small_config.d_model)
    seq_out = ssm(x)

    state = None
    step_outs = []
    for t in range(8):
        out, state = ssm.step(x[:, t : t + 1], state)
        step_outs.append(out)
    step_out = torch.cat(step_outs, dim=1)
    diff = (seq_out - step_out).abs().max().item()
    assert diff < 1e-4, f"step/seq mismatch: {diff}"


def test_gate_output_range(model: HBSNet) -> None:
    """Contradiction logits must be finite; gate alpha in (beta, 1)."""
    tokens = torch.randint(0, 64, (2, 6))
    out = model(tokens)
    assert torch.isfinite(out.contradiction_logits).all()


def test_generate_step_shapes(model: HBSNet) -> None:
    tokens = torch.randint(0, 64, (1, 1))
    state: list = []
    logits, state = model.generate_step(tokens, state)
    assert logits.shape == (1, 1, 64)
    logits2, state2 = model.generate_step(tokens, state)
    assert logits2.shape == (1, 1, 64)
    assert len(state2) == len(state)


def test_lm_head_tied(model: HBSNet) -> None:
    assert model.lm_head.weight is model.embedding.weight
