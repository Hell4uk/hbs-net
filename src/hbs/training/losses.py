"""
Composite loss for HBS-NET: text CE + contradiction BCE + energy penalty.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hbs.config import TrainingConfig


def composite_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    contradiction_logits: torch.Tensor,
    contradiction_labels: torch.Tensor,
    energy: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L = L_text + aux_weight * L_contradiction + energy_weight * E.

    logits: (batch, seq, vocab); labels: (batch, seq) with -100 ignore.
    contradiction_logits: (batch, seq); contradiction_labels: (batch, seq).
    energy: scalar.
    Returns (total_loss, metrics_dict).
    """
    vocab = logits.shape[-1]
    text_loss = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # Contradiction BCE over non-padding positions. Padding positions
    # carry label 0 but tokens are PAD; we mask via labels != -100 where
    # the position is a real next-token target.
    mask = labels != -100
    if mask.any():
        contra_loss = F.binary_cross_entropy_with_logits(
            contradiction_logits[mask].float(),
            contradiction_labels[mask].float(),
        )
    else:
        contra_loss = contradiction_logits.sum() * 0.0

    total = (
        text_loss
        + config.aux_weight * contra_loss
        + config.energy_weight * energy
    )

    with torch.no_grad():
        preds = (contradiction_logits > 0).long()
        correct = ((preds == contradiction_labels) & mask).sum().item()
        total_mask = mask.sum().item()
        acc = correct / max(total_mask, 1)
        tp = ((preds == 1) & (contradiction_labels == 1) & mask).sum().item()
        fp = ((preds == 1) & (contradiction_labels == 0) & mask).sum().item()
        fn = ((preds == 0) & (contradiction_labels == 1) & mask).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "loss": total.item(),
        "text_loss": text_loss.item(),
        "contra_loss": contra_loss.item(),
        "energy": energy.item(),
        "contra_acc": acc,
        "contra_f1": f1,
    }
    return total, metrics
