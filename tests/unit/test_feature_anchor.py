"""Unit tests for frozen-teacher feature anchoring."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from noisyclip.data.records import Batch
from noisyclip.losses.feature_anchor import FeatureAnchorLoss
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


def _batch() -> Batch:
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    return Batch(
        sample_ids=["a", "b"],
        image_weak=images,
        image_strong=None,
        targets=None,
        class_ids=None,
    )


def _output(embedding: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=torch.zeros((embedding.shape[0], 2), dtype=embedding.dtype),
        embedding=embedding,
        temperature=None,
    )


def test_feature_anchor_detaches_teacher_embedding() -> None:
    """Cosine anchor sends gradients into student embeddings only."""

    student = F.normalize(torch.tensor([[1.0, 1.0], [1.0, 0.0]]), dim=1).requires_grad_()
    teacher = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1).requires_grad_()

    value = FeatureAnchorLoss()(_batch(), _output(student), teacher, [_state("a"), _state("b")])
    value.backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert torch.isfinite(value)


def test_feature_anchor_requires_teacher_when_enabled() -> None:
    """Enabled feature anchoring fails fast if the frozen teacher is absent."""

    student = F.normalize(torch.ones((2, 2)), dim=1)

    with pytest.raises(ValueError, match="teacher_embedding"):
        FeatureAnchorLoss()(_batch(), _output(student), None, [_state("a"), _state("b")])


def test_feature_anchor_rejects_unnormalized_embeddings() -> None:
    """Embedding norm checks catch malformed model or teacher outputs."""

    teacher = F.normalize(torch.ones((2, 2)), dim=1)

    with pytest.raises(ValueError, match="L2-normalized"):
        FeatureAnchorLoss()(
            _batch(), _output(torch.ones((2, 2))), teacher, [_state("a"), _state("b")]
        )
