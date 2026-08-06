"""Unit tests for class-wise normalization and trust aggregation."""

from __future__ import annotations

import pytest
import torch

from noisyclip.data.records import SampleRecord
from noisyclip.noise.normalize import percentile_rank_by_class
from noisyclip.noise.state import SampleState
from noisyclip.noise.trust import ClasswiseTrustAggregator


def _record(sample_id: str, target: int) -> SampleRecord:
    return SampleRecord(
        sample_id, f"{target:04d}/x.jpg", "train", f"{target:04d}", target, None, 224, 224, True
    )


def _state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id, 1, 0.2, None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, "uncertain", None, None, 0
    )


def test_percentile_rank_is_independent_per_class_scale() -> None:
    """Class-wise rank is unaffected by another class's raw value scale."""

    values = torch.tensor([1.0, 2.0, 1000.0, 2000.0])
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    ranks = percentile_rank_by_class(values, targets, num_classes=2)

    assert torch.allclose(ranks, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_percentile_rank_handles_single_constant_and_nan_values() -> None:
    """Single-sample and constant classes are neutral; NaN becomes worst rank."""

    values = torch.tensor([5.0, 5.0, float("nan"), 9.0])
    targets = torch.tensor([0, 0, 1, 2], dtype=torch.int64)

    ranks = percentile_rank_by_class(values, targets, num_classes=3)

    assert torch.allclose(ranks, torch.tensor([0.5, 0.5, 0.0, 0.5]))
    assert torch.isfinite(ranks).all()


def test_trust_aggregation_outputs_scores_and_weights_in_range() -> None:
    """Enabled signals aggregate to continuous finite scores and weights."""

    records = [_record("a", 0), _record("b", 0), _record("c", 1), _record("d", 1)]
    previous = [_state(record.sample_id) for record in records]
    aggregator = ClasswiseTrustAggregator(
        {
            "ema_loss": 1.0,
            "prototype_similarity": 2.0,
            "augmentation_agreement": 0.0,
            "prototype_margin": 0.0,
            "prediction_stability": 0.0,
        },
        supervised_weight_min=0.1,
        supervised_weight_max=0.9,
    )

    updated = aggregator.update_epoch(
        records,
        {
            "ema_loss": torch.tensor([0.9, 0.1, 9.0, 1.0]),
            "prototype_similarity": torch.tensor([0.1, 0.9, 10.0, 20.0]),
        },
        previous,
        epoch=1,
    )

    scores = torch.tensor([state.trust_score for state in updated])
    weights = torch.tensor([state.supervised_weight for state in updated])
    assert torch.isfinite(scores).all()
    assert torch.isfinite(weights).all()
    assert bool(((0.0 <= scores) & (scores <= 1.0)).all())
    assert bool(((0.0 <= weights) & (weights <= 1.0)).all())
    assert len(set(weights.tolist())) > 1


def test_trust_aggregation_rejects_zero_coefficients_and_missing_signal() -> None:
    """Zero coefficient sums and absent enabled signal tensors fail."""

    with pytest.raises(ValueError, match="positive"):
        ClasswiseTrustAggregator({"ema_loss": 0.0})

    aggregator = ClasswiseTrustAggregator({"ema_loss": 1.0})
    records = [_record("a", 0)]

    with pytest.raises(ValueError, match="missing"):
        aggregator.update_epoch(records, {}, [_state("a")], epoch=0)
