"""Optional calibration metrics for validation logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.metrics.classification import MetricValue


def brier_score(logits: Tensor, targets: Tensor, *, num_classes: int) -> MetricValue:
    """Compute multiclass Brier score normalized to `[0, 1]` lower-is-better.

    Args:
        logits: Finite floating logits `[N, C]`.
        targets: Int64 targets `[N]`.
        num_classes: Positive class count `C`.

    Returns:
        Mean squared probability error divided by `2`, or `None` when no
        samples are present.

    Raises:
        ValueError: If shapes or class ranges are invalid.
    """

    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [N, {num_classes}].")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [N].")
    if logits.shape[0] == 0:
        return MetricValue(None, "no samples")
    if bool((targets < 0).any()) or bool((targets >= num_classes).any()):
        raise ValueError("targets out of range.")
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return MetricValue(float(((probabilities - one_hot).pow(2).sum(dim=1).mean() / 2.0).item()))


def expected_calibration_error(
    logits: Tensor,
    targets: Tensor,
    *,
    num_bins: int = 15,
) -> MetricValue:
    """Compute ECE over confidence bins as a proportion in `[0, 1]`.

    Args:
        logits: Finite floating logits `[N, C]`.
        targets: Int64 targets `[N]`.
        num_bins: Positive number of confidence bins.

    Returns:
        Expected calibration error, or `None` when no samples are present.

    Raises:
        ValueError: If shapes, finite checks, or bin count are invalid.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [N, C] and targets [N].")
    if logits.shape[0] == 0:
        return MetricValue(None, "no samples")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains NaN or Inf values.")
    probabilities = torch.softmax(logits.float(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == targets).float()
    ece = logits.new_zeros((), dtype=torch.float32)
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece = ece + mask.float().mean() * torch.abs(
                confidences[mask].mean() - correctness[mask].mean()
            )
    return MetricValue(float(ece.item()))
