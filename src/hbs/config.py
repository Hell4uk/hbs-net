"""
Main Centralized Project Configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    vocab_size: int = 32_000
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]")

@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 256
    d_state: int = 256
    n_layers: int = 6
    n_ssm_layers: int = 5
    n_attention_layers: int = 1
    n_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024
    contradiction_beta: float = 0.2

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    aux_weight: float = 0.5
    energy_weight: float = 0.1
    seed: int = 42

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data")

def load_config(path: str | Path) -> ProjectConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    tokenizer = TokenizerConfig(**raw.get("tokenizer", {}))
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))

    return ProjectConfig(
        tokenizer=tokenizer,
        model=model,
        training=training,
        checkpoint_dir=Path(raw.get("checkpoint_dir", "checkpoints")),
        data_dir=Path(raw.get("data_dir", "data")),
    )   