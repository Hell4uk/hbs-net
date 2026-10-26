"""Slow smoke tests: tiny training run and generation."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hbs.config import ProjectConfig, TrainingConfig
from hbs.inference import Generator
from hbs.tokenizer import BPETokenizer, build_corpus
from hbs.training import train
from hbs.training.trainer import save_checkpoint


@pytest.fixture(scope="module")
def tiny_config() -> ProjectConfig:
    return ProjectConfig(
        training=TrainingConfig(
            batch_size=4,
            learning_rate=3e-4,
            max_epochs=2,
            warmup_steps=2,
            aux_weight=0.5,
            energy_weight=0.1,
            seed=42,
        ),
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, tiny_config: ProjectConfig):
    tok = BPETokenizer(vocab_size=200)
    tok.train(build_corpus(200))
    config = replace(
        tiny_config,
        model=replace(tiny_config.model, vocab_size=tok.vocab_size, d_model=32, d_state=16, n_ssm_layers=2),
    )
    model, history = train(
        config, tok, device=torch.device("cpu"), n_samples=16
    )
    return model, tok, config, history


@pytest.mark.slow
def test_loss_decreases(trained) -> None:
    _, _, _, history = trained
    train_losses = [e["loss"] for e in history["train"]]
    assert len(train_losses) == 2
    assert train_losses[-1] < train_losses[0], (
        f"loss did not decrease: {train_losses}"
    )


@pytest.mark.slow
def test_checkpoint_roundtrip(trained, tmp_path: Path) -> None:
    model, tok, config, _ = trained
    path = tmp_path / "ckpt.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    save_checkpoint(path, model, optimizer, tok, config, epoch=2)
    assert path.exists()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_state" in checkpoint
    assert Path(checkpoint["tokenizer_path"]).exists()


@pytest.mark.slow
def test_generator_produces_text(trained) -> None:
    model, tok, _, _ = trained
    generator = Generator(model, tok)
    result = generator.generate(
        "трава была зелёная. трава оказалась жёлтая. Какая трава?",
        max_new_tokens=8,
        temperature=0.8,
    )
    assert isinstance(result.text, str)
    assert len(result.tokens) > 0
    assert 0.0 <= result.mean_contradiction <= 1.0
