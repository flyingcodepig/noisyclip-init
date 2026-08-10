"""Unit tests for F02 validation metrics."""

from __future__ import annotations

import pytest
import torch

from noisyclip.metrics.classification import compute_classification_metrics
from noisyclip.metrics.drift import feature_cosine_to_base
from noisyclip.metrics.robustness import augmentation_agreement, trusted_subset_top1


def test_classification_metrics_use_none_for_missing_classes() -> None:
    """Missing classes report `None` and do not masquerade as zero accuracy."""

    logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = torch.tensor([0, 1], dtype=torch.int64)
    metrics = compute_classification_metrics(logits, targets, num_classes=3)
    assert metrics.top1.value == 1.0
    assert metrics.per_class_accuracy[2].value is None
    assert metrics.per_class_accuracy[2].reason == "class has no validation samples"
    assert metrics.macro_accuracy.value == 1.0


def test_metric_guards_reject_invalid_shapes_and_ranges() -> None:
    """Classification metrics fail fast on malformed tensors."""

    logits = torch.tensor([[1.0, float("nan")]])
    targets = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_classification_metrics(logits, targets, num_classes=2)

    with pytest.raises(ValueError, match="trusted_mask"):
        trusted_subset_top1(
            torch.zeros((2, 2)),
            torch.tensor([0, 1]),
            torch.tensor([True]),
        )


def test_trusted_and_augmentation_metrics_return_nullable_reasons() -> None:
    """Unavailable subset/strong-view metrics return explicit reasons."""

    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    assert trusted_subset_top1(logits, targets, None).value is None
    assert augmentation_agreement(logits, None).reason == "strong-view logits unavailable"
    assert augmentation_agreement(logits, logits).value == 1.0


def test_feature_drift_reports_raw_cosine_and_mapped_alignment_separately() -> None:
    """Metric names expose raw cosine without losing the legacy alignment score."""

    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    base = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    drift = feature_cosine_to_base(student, base)

    assert drift.cosine == pytest.approx(0.0)
    assert drift.alignment == pytest.approx(0.5)

    unavailable = feature_cosine_to_base(student, None)
    assert unavailable.cosine is None
    assert unavailable.alignment is None
    assert unavailable.reason == "base CLIP embeddings unavailable"
