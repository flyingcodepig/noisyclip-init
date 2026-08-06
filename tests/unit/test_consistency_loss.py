"""Unit tests for prediction consistency loss."""

from __future__ import annotations

import pytest
import torch

from noisyclip.data.records import Batch
from noisyclip.losses.consistency import ConsistencyLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


def _state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=1,
        ema_loss=0.0,
        ema_probs=None,
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=1.0,
        prototype_margin=0.5,
        trust_score=1.0,
        supervised_weight=1.0,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )


def _batch(strong: bool) -> Batch:
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    return Batch(
        sample_ids=["a", "b"],
        image_weak=images,
        image_strong=images.clone() if strong else None,
        targets=None,
        class_ids=None,
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=logits,
        embedding=torch.zeros((logits.shape[0], 2), dtype=logits.dtype),
        temperature=None,
    )


def test_consistency_detaches_weak_target_branch() -> None:
    """KL target comes from weak logits without sending gradients into them."""

    weak_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    strong_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)

    loss = ConsistencyLoss(temperature=2.0, start_epoch=0)(
        _batch(strong=True),
        _output(weak_logits),
        _output(strong_logits),
        [_state("a"), _state("b")],
        epoch=0,
    )
    loss.backward()

    assert weak_logits.grad is None
    assert strong_logits.grad is not None
    assert torch.isfinite(loss)


def test_consistency_requires_strong_view_when_enabled() -> None:
    """Missing strong predictions fail even if the loss would be in warmup."""

    with pytest.raises(ValueError, match="student_strong"):
        ConsistencyLoss(start_epoch=10)(
            _batch(strong=False),
            _output(torch.zeros((2, 2))),
            None,
            [_state("a"), _state("b")],
            epoch=0,
        )


def test_consistency_disabled_does_not_require_strong_view() -> None:
    """Disabled consistency returns zero without touching the strong branch."""

    value = ConsistencyLoss(enabled=False)(
        _batch(strong=False),
        _output(torch.zeros((2, 2))),
        None,
        [_state("a"), _state("b")],
        epoch=0,
    )

    assert value.item() == 0.0
