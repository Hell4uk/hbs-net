"""
Entry point for training model
"""

from __future__ import annotations

import argparse
from pathlib import Path
from src.hbs.config import load_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HBS model")
    parser.add_argument('--config', type=Path, default='config/base.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Model: d_model={config.model.d_model}, layers={config.model.n_layers}")
    print(f"Training: batch_size={config.training.batch_size}, lr={config.training.learning_rate}")
    print("Training pipeline not yet implemented.")

    # TODO : Create logic for training model


if __name__ == "__main__":
    main()