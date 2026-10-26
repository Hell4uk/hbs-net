"""
Torch dataset and collate function for the synthetic contradiction data.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from hbs.tokenizer.base import BaseTokenizer
from hbs.dataset.synthetic import ContradictionSample, build_samples, labels_for_tokens

IGNORE_INDEX = -100


class SyntheticContradictionDataset(Dataset):
    """Tokenized contradiction samples with per-token labels.

    Each item: (tokens, labels, contradiction_labels) where
    - tokens: [BOS] + text tokens + [EOS]
    - labels: shifted next-token targets (IGNORE_INDEX on non-answer
      positions is applied later by the loss, here labels are the plain
      next-token targets for every position)
    - contradiction_labels: 0/1 per token, 1 on contradicting evidence.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.samples = build_samples(n_samples, seed=seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: ContradictionSample = self.samples[idx]
        tokens = self.tokenizer.encode(sample.text)
        tokens = [self.tokenizer.bos_id] + tokens + [self.tokenizer.eos_id]

        # Token-level contradiction labels aligned with `tokens`.
        # The tokenizer must expose token character spans; BPETokenizer
        # provides encode_with_spans for this.
        if hasattr(self.tokenizer, "encode_with_spans"):
            _, spans = self.tokenizer.encode_with_spans(sample.text)
        else:
            spans = [(0, len(sample.text))] * (len(tokens) - 2)
        char_labels = sample.contradiction_labels
        token_labels = labels_for_tokens(sample.text, char_labels, spans)
        # BOS gets 0, EOS gets 0.
        contradiction = [0] + token_labels + [0]

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "contradiction_labels": torch.tensor(contradiction, dtype=torch.long),
        }


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad batch to the max length in the batch.

    tokens are padded with PAD id; contradiction labels with 0; a
    labels tensor (next-token targets) is built with IGNORE_INDEX on
    padding and on the first position.
    """
    max_len = max(item["tokens"].shape[0] for item in batch)
    tokens = torch.full((len(batch), max_len), 0, dtype=torch.long)
    contradiction = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        tokens[i, :n] = item["tokens"]
        contradiction[i, :n] = item["contradiction_labels"]
        # Next-token targets: labels[t] = tokens[t+1].
        labels[i, : n - 1] = item["tokens"][1:]
    return {
        "tokens": tokens,
        "contradiction_labels": contradiction,
        "labels": labels,
    }
