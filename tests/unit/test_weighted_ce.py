"""Unit tests for weighted cross-entropy loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from noisyclip.data.records import Batch
from noisyclip.losses.weighted_ce import WeightedCrossEntropyLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


def _state(sample_id: str, weight: float) -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=1,
        ema_loss=0.0,
        ema_probs=None,
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=1.0,
        prototype_margin=0.5,
        trust_score=weight,
        supervised_weight=weight,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )


def _batch(targets: torch.Tensor) -> Batch:
    images = torch.zeros((3, 3, 224, 224), dtype=torch.float32)
    return Batch(
        sample_ids=["a", "b", "c"],
        image_weak=images,
        image_strong=None,
        targets=targets,
        class_ids=["0001", "0002", "0003"],
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=logits,
        embedding=torch.zeros((logits.shape[0], 4), dtype=logits.dtype),
        temperature=None,
    )


def test_weighted_ce_uses_weight_sum_normalization_and_returns_per_sample() -> None:
    """Weighted CE keeps gradient scale tied to effective weight, not batch size."""

    logits = torch.tensor(
        [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.5]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.int64)
    states = [_state("a", 1.0), _state("b", 0.25), _state("c", 0.0)]

    loss, per_sample = WeightedCrossEntropyLoss(label_smoothing=0.1)(
        _batch(targets),
        _output(logits),
        states,
    )

    expected_per_sample = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=0.1,
    )
    weights = torch.tensor([1.0, 0.25, 0.0])
    expected = (expected_per_sample * weights).sum() / weights.sum()

    assert torch.allclose(per_sample, expected_per_sample)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_weighted_ce_rejects_all_zero_weights() -> None:
    """All-zero supervised weights fail explicitly instead of hiding the batch."""

    logits = torch.zeros((3, 3), dtype=torch.float32)
    states = [_state("a", 0.0), _state("b", 0.0), _state("c", 0.0)]

    with pytest.raises(ValueError, match="positive"):
        WeightedCrossEntropyLoss()(
            _batch(torch.tensor([0, 1, 2], dtype=torch.int64)), _output(logits), states
        )


def test_weighted_ce_rejects_target_out_of_range_length_mismatch_and_nan() -> None:
    """Bad target shape/range and non-finite logits fail fast."""

    loss = WeightedCrossEntropyLoss()
    states = [_state("a", 1.0), _state("b", 1.0), _state("c", 1.0)]

    with pytest.raises(ValueError, match="range"):
        loss(
            _batch(torch.tensor([0, 1, 3], dtype=torch.int64)),
            _output(torch.zeros((3, 3), dtype=torch.float32)),
            states,
        )

    with pytest.raises(ValueError, match="shape"):
        loss(
            _batch(torch.tensor([0, 1], dtype=torch.int64)),
            _output(torch.zeros((3, 3), dtype=torch.float32)),
            states,
        )

    bad_logits = torch.zeros((3, 3), dtype=torch.float32)
    bad_logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        loss(_batch(torch.tensor([0, 1, 2], dtype=torch.int64)), _output(bad_logits), states)


def test_weighted_ce_rejects_misaligned_state_ids() -> None:
    """SampleState rows must correspond to the same sample IDs as the batch."""

    states = [_state("a", 1.0), _state("c", 1.0), _state("b", 1.0)]

    with pytest.raises(ValueError, match="ordered"):
        WeightedCrossEntropyLoss()(
            _batch(torch.tensor([0, 1, 2], dtype=torch.int64)),
            _output(torch.zeros((3, 3), dtype=torch.float32)),
            states,
        )
