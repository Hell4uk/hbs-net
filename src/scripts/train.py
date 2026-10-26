"""
Entry point for training the HBS model.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from hbs.config import ProjectConfig, load_config
from hbs.tokenizer import BPETokenizer, build_corpus
from hbs.training import save_checkpoint, train


def _prepare_tokenizer(config: ProjectConfig, n_corpus: int) -> BPETokenizer:
    """Load the tokenizer from data/ or train a fresh one on the corpus."""
    tokenizer_path = config.data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        return BPETokenizer.load(tokenizer_path)
    print("Training new BPE tokenizer on synthetic corpus...")
    tokenizer = BPETokenizer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
    )
    tokenizer.train(build_corpus(n_corpus))
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path} (vocab {tokenizer.vocab_size})")
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HBS model")
    parser.add_argument("--config", type=Path, default="config/base.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs from the config")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g. cpu or cuda")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of synthetic training samples")
    parser.add_argument("--n-corpus", type=int, default=4000,
                        help="Corpus size for tokenizer training")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config = replace(
            config,
            training=replace(config.training, max_epochs=args.epochs),
        )

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(
        f"Model: d_model={config.model.d_model}, "
        f"ssm_layers={config.model.n_ssm_layers}, "
        f"attention_layers={config.model.n_attention_layers}"
    )
    print(
        f"Training: batch_size={config.training.batch_size}, "
        f"lr={config.training.learning_rate}, "
        f"epochs={config.training.max_epochs}"
    )

    tokenizer = _prepare_tokenizer(config, args.n_corpus)

    # The model vocab must match the actual tokenizer vocab.
    config = replace(config, model=replace(
        config.model, vocab_size=tokenizer.vocab_size
    ))

    model, history = train(
        config, tokenizer, device=device, n_samples=args.n_samples
    )

    checkpoint_path = config.checkpoint_dir / "hbs_final.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    save_checkpoint(
        checkpoint_path, model, optimizer, tokenizer, config,
        epoch=config.training.max_epochs,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
