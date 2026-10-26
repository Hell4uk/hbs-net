"""Tests for the synthetic contradiction dataset."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hbs.dataset import IGNORE_INDEX, SyntheticContradictionDataset, collate_fn
from hbs.dataset.synthetic import build_samples
from hbs.tokenizer import BPETokenizer, build_corpus


def _tokenizer() -> BPETokenizer:
    tok = BPETokenizer(vocab_size=300)
    tok.train(build_corpus(300))
    return tok


def test_build_samples_deterministic() -> None:
    a = build_samples(20, seed=5)
    b = build_samples(20, seed=5)
    assert [s.text for s in a] == [s.text for s in b]
    assert [s.contradiction_labels for s in a] == [s.contradiction_labels for s in b]


def test_labels_mark_contradicting_sentences() -> None:
    samples = build_samples(50, seed=7)
    with_contra = [s for s in samples if s.has_contradiction]
    assert with_contra, "expected at least one contradicting sample"
    for sample in with_contra:
        assert any(sample.contradiction_labels), "labels must be non-empty"
        # Labels only cover evidence spans, not the whole text.
        assert sum(sample.contradiction_labels) < len(sample.contradiction_labels)


def test_dataset_item_shapes() -> None:
    tok = _tokenizer()
    ds = SyntheticContradictionDataset(tok, n_samples=8, seed=42)
    item = ds[0]
    assert item["tokens"].dim() == 1
    assert item["contradiction_labels"].shape == item["tokens"].shape
    # BOS/EOS are never contradictions.
    assert item["contradiction_labels"][0].item() == 0
    assert item["contradiction_labels"][-1].item() == 0


def test_collate_pads_and_shifts() -> None:
    tok = _tokenizer()
    ds = SyntheticContradictionDataset(tok, n_samples=8, seed=42)
    batch = collate_fn([ds[i] for i in range(4)])
    tokens, labels = batch["tokens"], batch["labels"]
    assert tokens.shape == labels.shape
    assert batch["contradiction_labels"].shape == tokens.shape
    # Next-token shift: labels[t] == tokens[t+1] on non-ignored positions.
    for i in range(4):
        n = (labels[i] != IGNORE_INDEX).sum().item() + 1
        assert torch.equal(labels[i, : n - 1], tokens[i, 1:n])
        assert (labels[i, n:] == IGNORE_INDEX).all()
