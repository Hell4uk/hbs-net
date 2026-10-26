"""
Training loop for HBS-NET: AdamW, warmup + cosine schedule, gradient
clipping, validation split, and checkpointing.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from hbs.config import ProjectConfig
from hbs.dataset import SyntheticContradictionDataset, collate_fn
from hbs.model import HBSNet
from hbs.tokenizer import BPETokenizer
from hbs.training.losses import composite_loss


@dataclass
class TrainMetrics:
    """Running metrics over an epoch."""

    loss: float = 0.0
    text_loss: float = 0.0
    contra_loss: float = 0.0
    energy: float = 0.0
    contra_acc: float = 0.0
    contra_f1: float = 0.0
    steps: int = 0

    def update(self, metrics: dict[str, float]) -> None:
        self.steps += 1
        for key in ("loss", "text_loss", "contra_loss", "energy", "contra_acc", "contra_f1"):
            setattr(self, key, getattr(self, key) + metrics[key])

    def averages(self) -> dict[str, float]:
        n = max(self.steps, 1)
        return {k: getattr(self, k) / n for k in
                ("loss", "text_loss", "contra_loss", "energy", "contra_acc", "contra_f1")}


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(
    config: ProjectConfig,
    tokenizer: BPETokenizer,
    device: torch.device | None = None,
    n_samples: int = 2000,
    val_fraction: float = 0.1,
    log_every: int = 20,
    micro_batch_size: int = 8,
) -> tuple[HBSNet, dict[str, list[dict[str, float]]]]:
    """Train HBSNet on synthetic contradiction data.

    Returns (model, history) where history has "train" and "val" lists
    of per-epoch average metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.training.seed)

    model = HBSNet(config.model).to(device)
    dataset = SyntheticContradictionDataset(
        tokenizer, n_samples=n_samples, seed=config.training.seed
    )
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.training.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True,
        collate_fn=collate_fn, generator=torch.Generator().manual_seed(config.training.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = max(len(train_loader) * config.training.max_epochs, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step, config.training.warmup_steps, total_steps
        ),
    )

    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(config.training.max_epochs):
        model.train()
        epoch_metrics = TrainMetrics()
        start = time.time()
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            contra = batch["contradiction_labels"].to(device)

            optimizer.zero_grad()
            # Gradient accumulation: the (batch, seq, d_model, d_state)
            # SSM intermediates are large, so process micro-batches.
            n = tokens.shape[0]
            n_micro = max(1, (n + micro_batch_size - 1) // micro_batch_size)
            epoch_metrics_batch = TrainMetrics()
            for i in range(n_micro):
                lo = i * micro_batch_size
                hi = min((i + 1) * micro_batch_size, n)
                output = model(tokens[lo:hi])
                loss, metrics = composite_loss(
                    output.logits, labels[lo:hi],
                    output.contradiction_logits, contra[lo:hi],
                    output.energy, config.training,
                )
                (loss / n_micro).backward()
                epoch_metrics_batch.update(metrics)
            epoch_metrics.update(epoch_metrics_batch.averages())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            epoch_metrics.update(metrics)

        val_metrics = evaluate(model, val_loader, config, device)
        history["train"].append(epoch_metrics.averages())
        history["val"].append(val_metrics)

        elapsed = time.time() - start
        print(
            f"epoch {epoch + 1}/{config.training.max_epochs} "
            f"({elapsed:.1f}s) "
            f"train_loss={history['train'][-1]['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_contra_acc={val_metrics['contra_acc']:.3f} "
            f"val_contra_f1={val_metrics['contra_f1']:.3f}"
        )

    return model, history


@torch.no_grad()
def evaluate(
    model: HBSNet,
    loader: DataLoader,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = TrainMetrics()
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        contra = batch["contradiction_labels"].to(device)
        n = tokens.shape[0]
        for i in range(0, n, 8):
            lo, hi = i, min(i + 8, n)
            output = model(tokens[lo:hi])
            _, batch_metrics = composite_loss(
                output.logits, labels[lo:hi],
                output.contradiction_logits, contra[lo:hi],
                output.energy, config.training,
            )
            metrics.update(batch_metrics)
    return metrics.averages()


def save_checkpoint(
    path: Path,
    model: HBSNet,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    config: ProjectConfig,
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path.with_name("tokenizer.json")
    tokenizer.save(tokenizer_path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_path": str(tokenizer_path),
            "config": {"model": asdict(model.config)},
            "epoch": epoch,
        },
        path,
    )
