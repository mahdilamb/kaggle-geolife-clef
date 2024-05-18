"""Module containing metrics."""

import torch


def mean_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    """Calculate the mean accuracy."""
    return torch.mean((preds == targets).float(), dim=1).cpu().numpy().mean()
