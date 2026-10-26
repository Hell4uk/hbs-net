"""
Synthetic contradiction dataset: sequences of evidence sentences about
entity attributes, followed by a question. Some evidence sentences
contradict earlier ones; the answer is always the latest evidence.

Each sample provides per-token contradiction labels derived from ground
truth, so the DialecticalGate head can be supervised directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hbs.tokenizer.corpus import _ANSWER_PREFIXES, _ATTRIBUTES, _ENTITIES, _QUESTIONS, _TEMPLATES

_EVIDENCE_TEMPLATES = [
    "{entity} была {attribute}.",
    "{entity} оказалась {attribute}.",
    "Свидетель утверждает: {entity} {attribute}.",
    "По словам очевидца, {entity} стала {attribute}.",
    "Известно, что {entity} всегда была {attribute}.",
    "В отчёте сказано, что {entity} {attribute}.",
    "Раньше {entity} была {attribute}.",
    "Сейчас {entity} {attribute}.",
]


@dataclass(frozen=True)
class ContradictionSample:
    """One training example: evidence + question + answer + labels."""

    text: str
    answer: str
    contradiction_labels: list[int]
    has_contradiction: bool


def _attribute_pool(rng: random.Random) -> list[str]:
    return rng.choice(list(_ATTRIBUTES.values()))


def build_samples(
    n_samples: int,
    seed: int = 42,
    min_evidence: int = 2,
    max_evidence: int = 5,
    contradiction_rate: float = 0.5,
) -> list[ContradictionSample]:
    """Generate deterministic contradiction samples.

    Each sample: 2-5 evidence sentences about one entity, a question,
    and the answer (the latest evidence attribute). When a contradiction
    occurs, the contradicting sentence's tokens are labeled 1.
    """
    rng = random.Random(seed)
    samples: list[ContradictionSample] = []

    for _ in range(n_samples):
        entity = rng.choice(_ENTITIES)
        pool = _attribute_pool(rng)
        n_evidence = rng.randint(min_evidence, max_evidence)
        attributes = [rng.choice(pool) for _ in range(n_evidence)]

        # Decide which evidence sentences contradict an earlier one.
        # First sentence can never contradict.
        contradicts = [False]
        for i in range(1, n_evidence):
            contradicts.append(
                attributes[i] != attributes[i - 1] and rng.random() < 0.7
            )
        # If no contradiction was drawn but the sample should have one,
        # force the last sentence to contradict the first.
        has_contradiction = any(contradicts)
        if not has_contradiction and rng.random() < contradiction_rate:
            attributes[-1] = rng.choice(
                [a for a in pool if a != attributes[0]]
            )
            contradicts[-1] = True
            has_contradiction = True

        evidence_sentences = [
            rng.choice(_EVIDENCE_TEMPLATES).format(
                entity=entity, attribute=attributes[i]
            )
            for i in range(n_evidence)
        ]
        question = rng.choice(_QUESTIONS).format(entity=entity)
        answer = rng.choice(_ANSWER_PREFIXES).format(
            entity=entity, attribute=attributes[-1]
        )

        parts = evidence_sentences + [question]
        text = " ".join(parts) + " " + answer

        # Per-token labels over the full text: 1 on tokens of contradicting
        # evidence sentences, 0 elsewhere. We mark by character spans.
        spans: list[tuple[int, int]] = []
        pos = 0
        for i, sentence in enumerate(evidence_sentences):
            end = pos + len(sentence)
            if contradicts[i]:
                spans.append((pos, end))
            pos = end + 1  # space
        # Question and answer are never labeled as contradictions.
        labels = _labels_from_spans(text, spans)

        samples.append(
            ContradictionSample(
                text=text,
                answer=answer,
                contradiction_labels=labels,
                has_contradiction=has_contradiction,
            )
        )
    return samples


def _labels_from_spans(text: str, spans: list[tuple[int, int]]) -> list[int]:
    """One label per character of text; 1 inside any span."""
    labels = [0] * len(text)
    for start, end in spans:
        for i in range(start, end):
            labels[i] = 1
    return labels


def labels_for_tokens(
    text: str, char_labels: list[int], token_spans: list[tuple[int, int]]
) -> list[int]:
    """Map character-level labels to token-level labels.

    A token is labeled 1 if any of its characters is labeled 1.
    token_spans: (start, end) character offsets per token.
    """
    token_labels = []
    for start, end in token_spans:
        if any(char_labels[start:end]):
            token_labels.append(1)
        else:
            token_labels.append(0)
    return token_labels
