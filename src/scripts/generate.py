"""
CLI entry point for generation from a trained checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hbs.config import ModelConfig
from hbs.inference import Generator
from hbs.model import HBSNet
from hbs.tokenizer import BPETokenizer


def load_generator(checkpoint_path: Path) -> Generator:
    """Load a Generator from a checkpoint saved by save_checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = HBSNet(model_config)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = BPETokenizer.load(Path(checkpoint["tokenizer_path"]))
    return Generator(model, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with a trained HBS model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--evidence", type=str, nargs="*", default=[],
                        help="Evidence sentences (quoted strings)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    generator = load_generator(args.checkpoint)

    parts = list(args.evidence) + [args.question]
    prompt = " ".join(parts)

    result = generator.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Ответ: {result.text}")
    print(f"Индекс противоречия: {result.mean_contradiction:.3f}")


if __name__ == "__main__":
    main()
