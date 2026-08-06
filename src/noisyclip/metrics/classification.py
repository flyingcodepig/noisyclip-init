"""Classification metrics with explicit missing-class behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Metric value with a nullable result and optional reason.

    Attributes:
        value: Proportion in `[0, 1]`, or `None` when not computable.
        reason: Human-readable reason when `value` is `None`.
    """

    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate nullable metric ranges."""

        if self.value is None:
            if not self.reason:
                raise ValueError("MetricValue with None value requires a reason.")
        elif not 0.0 <= self.value <= 1.0:
            raise ValueError(f"metric value must be in [0, 1], got {self.value}.")


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Top-level classification metrics and confusion matrix."""

    top1: MetricValue
    macro_accuracy: MetricValue
    bottom_quartile_accuracy: MetricValue
    per_class_accuracy: dict[int, MetricValue]
    confusion_matrix: Tensor


def confusion_matrix(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """Compute an integer `[C, C]` confusion matrix.

    Args:
        predictions: Int64 tensor `[N]` with values in `[0, C)`.
        targets: Int64 tensor `[N]` with values in `[0, C)`.
        num_classes: Positive number of classes.

    Returns:
        Int64 confusion matrix where rows are targets and columns predictions.

    Raises:
        TypeError: If tensors are not int64.
        ValueError: If shapes, ranges, or class count are invalid.
    """

    _validate_label_tensors(predictions, targets, num_classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        matrix[int(target), int(prediction)] += 1
    return matrix


def compute_classification_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute top-1, macro, per-class, and bottom-quartile accuracy.

    Args:
        logits: Finite floating tensor `[N, C]`.
        targets: Int64 tensor `[N]` with values in `[0, C)`.
        num_classes: Positive number of classes and second logits dimension.

    Returns:
        `ClassificationMetrics`; missing class accuracies are `None` with a
        reason, and macro/bottom-quartile use only present classes.

    Raises:
        TypeError: If `logits` is not floating or targets are not int64.
        ValueError: If shapes, ranges, or finite checks fail.
    """

    _validate_logits(logits, num_classes)
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape [N], got {tuple(targets.shape)}.")
    predictions = logits.argmax(dim=1).to(torch.int64)
    matrix = confusion_matrix(predictions, targets, num_classes)
    total = int(matrix.sum().item())
    top1 = (
        MetricValue(None, "no labeled samples")
        if total == 0
        else MetricValue(float(matrix.diag().sum().item() / total))
    )
    per_class = per_class_accuracy_from_confusion(matrix)
    present_values = [metric.value for metric in per_class.values() if metric.value is not None]
    if not present_values:
        macro = MetricValue(None, "no classes with validation samples")
        bottom = MetricValue(None, "no classes with validation samples")
    else:
        macro = MetricValue(float(sum(present_values) / len(present_values)))
        sorted_values = sorted(present_values)
        count = max(1, len(sorted_values) // 4)
        bottom = MetricValue(float(sum(sorted_values[:count]) / count))
    return ClassificationMetrics(
        top1=top1,
        macro_accuracy=macro,
        bottom_quartile_accuracy=bottom,
        per_class_accuracy=per_class,
        confusion_matrix=matrix,
    )


def per_class_accuracy_from_confusion(matrix: Tensor) -> dict[int, MetricValue]:
    """Return nullable per-class accuracies from a `[C, C]` confusion matrix.

    Args:
        matrix: Int64 square confusion matrix.

    Returns:
        Mapping from class index to `MetricValue`.

    Raises:
        TypeError: If matrix dtype is not int64.
        ValueError: If matrix is not square or contains negative counts.
    """

    if matrix.dtype != torch.int64:
        raise TypeError("confusion matrix must be int64.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square [C, C], got {tuple(matrix.shape)}.")
    if bool((matrix < 0).any()):
        raise ValueError("confusion matrix contains negative counts.")
    metrics: dict[int, MetricValue] = {}
    for class_index in range(matrix.shape[0]):
        support = int(matrix[class_index].sum().item())
        if support == 0:
            metrics[class_index] = MetricValue(None, "class has no validation samples")
        else:
            metrics[class_index] = MetricValue(
                float(matrix[class_index, class_index].item() / support)
            )
    return metrics


def _validate_logits(logits: Tensor, num_classes: int) -> None:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [N, {num_classes}], got {tuple(logits.shape)}.")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains NaN or Inf values.")


def _validate_label_tensors(predictions: Tensor, targets: Tensor, num_classes: int) -> None:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if predictions.dtype != torch.int64 or targets.dtype != torch.int64:
        raise TypeError("predictions and targets must be int64 tensors.")
    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have matching shape [N], got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
        )
    for name, tensor in (("predictions", predictions), ("targets", targets)):
        if tensor.numel() and (bool((tensor < 0).any()) or bool((tensor >= num_classes).any())):
            raise ValueError(f"{name} values must be in [0, {num_classes}).")
