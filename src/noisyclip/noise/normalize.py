"""Class-wise normalization for raw trust signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ClasswisePercentileNormalizer:
    """Normalize raw `[N]` values by percentile rank inside each class.

    Args:
        higher_is_better: When true, larger raw values receive higher ranks.
            When false, smaller raw values receive higher ranks.

    NaN values are assigned rank `0` for their class. A finite single-sample or
    constant-valued class receives neutral rank `0.5`. Outputs are finite and
    clipped to `[0, 1]`; no global thresholding is applied.

    Raises:
        TypeError: If `values` is not floating-point or `targets` is not int64.
        ValueError: If shapes, finite target ranges, or class ids are invalid.
    """

    higher_is_better: bool = True

    def __call__(self, values: Tensor, targets: Tensor, num_classes: int) -> Tensor:
        """Return class-wise percentile ranks shaped `[N]` in `[0, 1]`."""

        return percentile_rank_by_class(
            values,
            targets,
            num_classes,
            higher_is_better=self.higher_is_better,
        )


def percentile_rank_by_class(
    values: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    higher_is_better: bool = True,
) -> Tensor:
    """Rank-normalize `[N]` raw values independently within each target class.

    Args:
        values: Floating-point tensor shaped `[N]`. `NaN` is allowed and mapped
            to rank `0`; `Inf` is rejected.
        targets: Int64 tensor shaped `[N]` with values in `[0, C)`.
        num_classes: Positive class count `C`.
        higher_is_better: Whether larger finite raw values should receive
            larger normalized scores.

    Returns:
        A finite floating-point tensor shaped `[N]` with values in `[0, 1]`.

    Raises:
        TypeError: If input dtypes are invalid.
        ValueError: If shapes, class ranges, or infinite values are invalid.
    """

    _validate_rank_inputs(values, targets, num_classes)
    normalized = torch.zeros_like(values, dtype=torch.float32)
    working = values.detach().to(torch.float32)
    if not higher_is_better:
        working = -working
    for class_index in range(num_classes):
        mask = targets == class_index
        if not bool(mask.any()):
            continue
        class_values = working[mask]
        finite_mask = torch.isfinite(class_values)
        if not bool(finite_mask.any()):
            normalized[mask] = 0.0
            continue
        finite_values = class_values[finite_mask]
        finite_ranks = _rank_finite_values(finite_values)
        class_ranks = torch.zeros_like(class_values, dtype=torch.float32)
        class_ranks[finite_mask] = finite_ranks
        normalized[mask] = class_ranks
    return normalized.clamp(0.0, 1.0)


def _rank_finite_values(values: Tensor) -> Tensor:
    if values.numel() == 1 or bool(torch.all(values == values[0])):
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_ranks = torch.empty_like(sorted_values, dtype=torch.float32)
    denominator = float(values.numel() - 1)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        average_position = (start + end - 1) / 2.0
        sorted_ranks[start:end] = average_position / denominator
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _validate_rank_inputs(values: Tensor, targets: Tensor, num_classes: int) -> None:
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}.")
    if values.ndim != 1:
        raise ValueError(f"values must have shape [N], got {tuple(values.shape)}.")
    if not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor.")
    if torch.isinf(values).any():
        raise ValueError("values contains Inf; NaN is handled as worst rank.")
    if targets.ndim != 1 or targets.shape[0] != values.shape[0]:
        raise ValueError(f"targets must have shape [N], got {tuple(targets.shape)}.")
    if targets.dtype != torch.int64:
        raise TypeError("targets must be an int64 tensor.")
    if bool((targets < 0).any()) or bool((targets >= num_classes).any()):
        raise ValueError(f"targets must be in [0, {num_classes}).")
