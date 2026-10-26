"""Tests for the BPE tokenizer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hbs.tokenizer import BPETokenizer, build_corpus


@pytest.fixture(scope="module")
def trained_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(build_corpus(300))
    return tokenizer


def test_roundtrip(trained_tokenizer: BPETokenizer) -> None:
    text = "трава была зелёная"
    ids = trained_tokenizer.encode(text)
    assert ids
    assert trained_tokenizer.decode(ids) == text


def test_special_tokens(trained_tokenizer: BPETokenizer) -> None:
    assert trained_tokenizer.pad_id == 0
    assert trained_tokenizer.unk_id == 1
    assert trained_tokenizer.bos_id == 2
    assert trained_tokenizer.eos_id == 3
    assert trained_tokenizer.sep_id == 4


def test_save_load_roundtrip(trained_tokenizer: BPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    assert trained_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "небо оказалось синяя"
    assert loaded.encode(text) == trained_tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_train_determinism() -> None:
    corpus = build_corpus(200, seed=3)
    tok_a = BPETokenizer(vocab_size=200)
    tok_b = BPETokenizer(vocab_size=200)
    tok_a.train(corpus)
    tok_b.train(corpus)
    text = "море было холодная"
    assert tok_a.encode(text) == tok_b.encode(text)


def test_encode_with_spans(trained_tokenizer: BPETokenizer) -> None:
    text = "трава зелёная"
    ids, spans = trained_tokenizer.encode_with_spans(text)
    assert len(ids) == len(spans)
    for start, end in spans:
        assert 0 <= start < end <= len(text)
    # Reconstructing the text from spans must cover all non-space chars.
    covered = "".join(text[s:e] for s, e in spans)
    assert covered == text.replace(" ", "")
