"""
Autoregressive generation for HBS-NET with SSM state and KV-cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hbs.model import HBSNet
from hbs.tokenizer import BPETokenizer


@dataclass
class GenerationResult:
    """Result of one generation call."""

    text: str
    tokens: list[int]
    mean_contradiction: float


class Generator:
    """Autoregressive generator over HBSNet in step mode."""

    def __init__(
        self,
        model: HBSNet,
        tokenizer: BPETokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        stop_at_eos: bool = True,
    ) -> GenerationResult:
        """Generate a continuation of `prompt`.

        The prompt is processed token by token in step mode to build the
        SSM state and KV-cache, then new tokens are sampled until EOS or
        max_new_tokens.
        """
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        if not tokens:
            tokens = [self.tokenizer.bos_id]

        state: list[torch.Tensor | None] = []
        logits = None
        for token_id in tokens:
            token = torch.tensor(
                [[token_id]], dtype=torch.long, device=self.device
            )
            logits, state = self.model.generate_step(token, state)

        generated: list[int] = []
        contradiction_scores: list[float] = []

        for _ in range(max_new_tokens):
            if logits is None:
                break
            next_token = self._sample(logits[0, -1], temperature, top_k)
            if stop_at_eos and next_token == self.tokenizer.eos_id:
                break
            generated.append(next_token)
            token = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.device
            )
            logits, state = self.model.generate_step(token, state)

        text = self.tokenizer.decode(generated)
        mean_contra = (
            sum(contradiction_scores) / len(contradiction_scores)
            if contradiction_scores
            else 0.0
        )
        return GenerationResult(
            text=text, tokens=generated, mean_contradiction=mean_contra
        )

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> int:
        if temperature <= 0:
            return int(logits.argmax().item())
        logits = logits.float() / temperature
        if top_k > 0:
            k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, k).values[..., -1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())
