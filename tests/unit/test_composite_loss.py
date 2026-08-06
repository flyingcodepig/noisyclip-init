"""Unit tests for configured robust composite loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from noisyclip.config.schema import LossConfig
from noisyclip.data.records import Batch
from noisyclip.losses.composite import RobustCompositeLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


def _state(sample_id: str, weight: float = 1.0) -> SampleState:
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


def _batch(*, strong: bool = True, targets: torch.Tensor | None = None) -> Batch:
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    return Batch(
        sample_ids=["a", "b"],
        image_weak=images,
        image_strong=images.clone() if strong else None,
        targets=targets,
        class_ids=["0001", "0002"] if targets is not None else None,
    )


def _output(logits: torch.Tensor, embedding: torch.Tensor | None = None) -> ModelOutput:
    if embedding is None:
        embedding = F.normalize(torch.ones((logits.shape[0], 2), dtype=logits.dtype), dim=1)
    return ModelOutput(logits=logits, embedding=embedding, temperature=None)


def test_composite_independent_disable_and_component_names() -> None:
    """Disabled terms do not affect total; enabled terms use stable names."""

    config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": True, "label_smoothing": 0.0, "weight": 1.0},
            "elr": {"enabled": False},
            "consistency": {"enabled": False},
            "feature_anchor": {"enabled": False},
            "logit_adjustment": {"enabled": False},
        }
    )
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    result = RobustCompositeLoss(config)(
        _batch(targets=torch.tensor([0, 1], dtype=torch.int64)),
        _output(logits),
        None,
        None,
        [_state("a", 1.0), _state("b", 0.5)],
        epoch=0,
    )

    assert set(result.components) == {"loss/effective_supervised_weight", "loss/ce"}
    assert torch.allclose(result.total, result.components["loss/ce"])
    assert result.components["loss/effective_supervised_weight"].item() == pytest.approx(1.5)
    assert result.per_sample_supervised is not None
    assert not result.per_sample_supervised.requires_grad


def test_composite_all_modules_disabled_returns_explicit_zero() -> None:
    """The illegal-for-training all-disabled setup has explicit zero-loss behavior."""

    config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": False},
            "elr": {"enabled": False},
            "consistency": {"enabled": False},
            "feature_anchor": {"enabled": False},
            "logit_adjustment": {"enabled": False},
        }
    )
    result = RobustCompositeLoss(config)(
        _batch(strong=False, targets=None),
        _output(torch.zeros((2, 2))),
        None,
        None,
        [_state("a"), _state("b")],
        epoch=0,
    )

    assert result.total.item() == 0.0
    assert result.per_sample_supervised is None
    assert set(result.components) == {"loss/effective_supervised_weight"}


def test_composite_sums_enabled_weighted_terms() -> None:
    """CE, consistency, and feature anchor contribute only when enabled."""

    config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": True, "weight": 1.0},
            "elr": {"enabled": False},
            "consistency": {"enabled": True, "weight": 0.5, "start_epoch": 0},
            "feature_anchor": {"enabled": True, "weight": 2.0},
            "logit_adjustment": {"enabled": False},
        }
    )
    weak = _output(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    strong = _output(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
    teacher = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    result = RobustCompositeLoss(config)(
        _batch(targets=torch.tensor([0, 1], dtype=torch.int64)),
        weak,
        strong,
        teacher,
        [_state("a"), _state("b")],
        epoch=0,
    )

    expected = (
        result.components["loss/ce"]
        + result.components["loss/consistency"]
        + result.components["loss/feature_anchor"]
    )
    assert torch.allclose(result.total, expected)


def test_composite_rejects_nan_component_inputs() -> None:
    """Any NaN/Inf input that would produce a bad component aborts immediately."""

    config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": True},
            "elr": {"enabled": False},
            "consistency": {"enabled": False},
            "feature_anchor": {"enabled": False},
            "logit_adjustment": {"enabled": False},
        }
    )
    logits = torch.zeros((2, 2), dtype=torch.float32)
    logits[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        RobustCompositeLoss(config)(
            _batch(targets=torch.tensor([0, 1], dtype=torch.int64)),
            _output(logits),
            None,
            None,
            [_state("a"), _state("b")],
            epoch=0,
        )


def test_composite_requires_strong_and_teacher_for_enabled_terms() -> None:
    """Enabled optional modules fail clearly when their required inputs are absent."""

    consistency_config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": False},
            "elr": {"enabled": False},
            "consistency": {"enabled": True, "weight": 1.0, "start_epoch": 0},
            "feature_anchor": {"enabled": False},
            "logit_adjustment": {"enabled": False},
        }
    )
    with pytest.raises(ValueError, match="student_strong"):
        RobustCompositeLoss(consistency_config)(
            _batch(strong=False),
            _output(torch.zeros((2, 2))),
            None,
            None,
            [_state("a"), _state("b")],
            epoch=0,
        )

    anchor_config = LossConfig.model_validate(
        {
            "cross_entropy": {"enabled": False},
            "elr": {"enabled": False},
            "consistency": {"enabled": False},
            "feature_anchor": {"enabled": True, "weight": 1.0},
            "logit_adjustment": {"enabled": False},
        }
    )
    with pytest.raises(ValueError, match="teacher_embedding"):
        RobustCompositeLoss(anchor_config)(
            _batch(strong=False),
            _output(torch.zeros((2, 2))),
            None,
            None,
            [_state("a"), _state("b")],
            epoch=0,
        )
