"""
Byte-level BPE tokenizer with trainable merges, special tokens and
save/load support. Implements the BaseTokenizer contract.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hbs.tokenizer.base import BaseTokenizer

_PAD, _UNK, _BOS, _EOS, _SEP = "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"
_SPECIALS = (_PAD, _UNK, _BOS, _EOS, _SEP)


class BPETokenizer(BaseTokenizer):
    """Character-level BPE over pre-tokenized words.

    Words are split on whitespace; each word is encoded as a sequence of
    characters (with an end-of-word marker) and merged according to learned
    merge rules. Unknown tokens map to [UNK] only when a character was never
    seen during training; otherwise every character has an id.
    """

    def __init__(self, vocab_size: int = 32_000, min_frequency: int = 2) -> None:
        self._target_vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._vocab: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        for token in _SPECIALS:
            self._add_token(token)

    # ------------------------------------------------------------------ #
    # BaseTokenizer contract
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._vocab[_PAD]

    @property
    def bos_id(self) -> int:
        return self._vocab[_BOS]

    @property
    def eos_id(self) -> int:
        return self._vocab[_EOS]

    @property
    def sep_id(self) -> int:
        return self._vocab[_SEP]

    @property
    def unk_id(self) -> int:
        return self._vocab[_UNK]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for word in text.split():
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_with_spans(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode text and return character spans for each token.

        Spans are (start, end) offsets into `text`, one per token,
        excluding any BOS/EOS. Whitespace between words is not part of
        any span.
        """
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        pos = 0
        for word in text.split():
            start = text.find(word, pos)
            end = start + len(word)
            word_ids = self._encode_word(word)
            # Distribute the word span across its subword tokens.
            n = len(word_ids)
            if n == 1:
                spans.append((start, end))
            else:
                # Approximate: split the word evenly across subtokens.
                step = (end - start) / n
                for k in range(n):
                    s = start + int(round(k * step))
                    e = start + int(round((k + 1) * step))
                    spans.append((s, e))
            ids.extend(word_ids)
            pos = end
        return ids, spans

    def decode(self, ids: list[int]) -> str:
        inv = {i: t for t, i in self._vocab.items()}
        words: list[str] = []
        current: list[str] = []
        for i in ids:
            token = inv.get(i)
            if token is None or token in _SPECIALS:
                continue
            if token.endswith("</w>"):
                current.append(token[: -len("</w>")])
                words.append("".join(current))
                current = []
            else:
                current.append(token)
        if current:
            words.append("".join(current))
        return " ".join(words)

    def save(self, path: Path) -> bool:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self._target_vocab_size,
            "min_frequency": self._min_frequency,
            "vocab": self._vocab,
            "merges": [list(pair) for pair in self._merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return True

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(
            vocab_size=payload["vocab_size"],
            min_frequency=payload["min_frequency"],
        )
        tok._vocab = {t: int(i) for t, i in payload["vocab"].items()}
        tok._merges = [tuple(pair) for pair in payload["merges"]]
        return tok

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, texts: list[str]) -> None:
        """Learn BPE merges from texts until vocab_size is reached."""
        word_counts: Counter[str] = Counter()
        for text in texts:
            for word in text.split():
                word_counts[word] += 1

        # Initial alphabet: characters + end-of-word marker.
        sequences: dict[tuple[str, ...], int] = {}
        for word, count in word_counts.items():
            symbols = tuple(list(word) + ["</w>"])
            sequences[symbols] = sequences.get(symbols, 0) + count

        for symbols in sequences:
            for ch in symbols:
                self._add_token(ch)

        while self.vocab_size < self._target_vocab_size:
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break
            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )
            if best_count < self._min_frequency:
                break
            self._merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            self._add_token(merged)
            sequences = self._apply_merge(sequences, best_pair, merged)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _add_token(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    @staticmethod
    def _count_pairs(
        sequences: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, str], int]:
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, count in sequences.items():
            for a, b in zip(symbols, symbols[1:]):
                counts[(a, b)] += count
        return dict(counts)

    @staticmethod
    def _apply_merge(
        sequences: dict[tuple[str, ...], int],
        pair: tuple[str, str],
        merged: str,
    ) -> dict[tuple[str, ...], int]:
        out: dict[tuple[str, ...], int] = {}
        first, second = pair
        for symbols, count in sequences.items():
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            key = tuple(new_symbols)
            out[key] = out.get(key, 0) + count
        return out

    def _encode_word(self, word: str) -> list[int]:
        symbols: list[str] = list(word) + ["</w>"]
        for pair, merged in [(p, p[0] + p[1]) for p in self._merges]:
            if len(symbols) < 2:
                break
            i = 0
            while i < len(symbols) - 1:
                if (symbols[i], symbols[i + 1]) == pair:
                    symbols[i : i + 2] = [merged]
                else:
                    i += 1
        ids: list[int] = []
        for symbol in symbols:
            ids.append(self._vocab.get(symbol, self.unk_id))
        return ids
