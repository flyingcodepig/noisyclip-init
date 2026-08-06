"""Subset and consistency metrics for noisy-label validation."""

from __future__ import annotations

import torch
from torch import Tensor

from noisyclip.metrics.classification import MetricValue


def trusted_subset_top1(
    logits: Tensor,
    targets: Tensor,
    trusted_mask: Tensor | None,
) -> MetricValue:
    """Compute top-1 accuracy over a trusted subset.

    Args:
        logits: Finite floating logits `[N, C]`.
        targets: Int64 targets `[N]`.
        trusted_mask: Optional bool mask `[N]`; `None` means unavailable.

    Returns:
        Proportion in `[0, 1]`, or `None` with reason when no trusted samples
        exist or the mask is unavailable.

    Raises:
        TypeError: If mask dtype is not bool.
        ValueError: If shapes are inconsistent.
    """

    if trusted_mask is None:
        return MetricValue(None, "trusted mask unavailable")
    if trusted_mask.dtype != torch.bool:
        raise TypeError("trusted_mask must be bool.")
    if trusted_mask.ndim != 1 or trusted_mask.shape[0] != logits.shape[0]:
        raise ValueError("trusted_mask must have shape [N].")
    if int(trusted_mask.sum().item()) == 0:
        return MetricValue(None, "trusted subset is empty")
    predictions = logits.argmax(dim=1)
    return MetricValue(
        float((predictions[trusted_mask] == targets[trusted_mask]).float().mean().item())
    )


def augmentation_agreement(logits_weak: Tensor, logits_strong: Tensor | None) -> MetricValue:
    """Compute weak/strong argmax agreement.

    Args:
        logits_weak: Finite floating logits `[N, C]`.
        logits_strong: Optional finite floating logits `[N, C]`.

    Returns:
        Agreement in `[0, 1]`, or `None` with reason when strong logits are
        unavailable.

    Raises:
        ValueError: If shapes differ or logits are non-finite.
    """

    if logits_strong is None:
        return MetricValue(None, "strong-view logits unavailable")
    if logits_weak.shape != logits_strong.shape or logits_weak.ndim != 2:
        raise ValueError("weak and strong logits must have matching shape [N, C].")
    if not torch.isfinite(logits_weak).all() or not torch.isfinite(logits_strong).all():
        raise ValueError("augmentation logits contain NaN or Inf values.")
    if logits_weak.shape[0] == 0:
        return MetricValue(None, "no samples")
    agreement = (logits_weak.argmax(dim=1) == logits_strong.argmax(dim=1)).float().mean()
    return MetricValue(float(agreement.item()))
