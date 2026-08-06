"""Tests for linear and cosine classifier heads."""

from __future__ import annotations

import pytest
import torch

from noisyclip.models.classifier import (
    CosineClassifierHead,
    LinearClassifierHead,
    build_classifier_head,
)


def test_linear_head_outputs_logits_and_gradients() -> None:
    """Linear head maps `[B,D]` embeddings to unsoftmaxed `[B,C]` logits."""

    head = LinearClassifierHead(embedding_dim=4, num_classes=3)
    embeddings = torch.randn(5, 4, requires_grad=True)

    logits = head(embeddings)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (5, 3)
    assert embeddings.grad is not None
    assert head.linear.weight.grad is not None


def test_cosine_head_normalizes_inputs_weights_and_uses_temperature() -> None:
    """Cosine logits equal normalized dot products multiplied by temperature."""

    head = CosineClassifierHead(
        embedding_dim=2,
        num_classes=2,
        temperature_init=2.0,
        temperature_min=0.5,
        temperature_max=4.0,
    )
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
    embeddings = torch.tensor([[10.0, 0.0], [0.0, 7.0]], requires_grad=True)

    logits = head(embeddings)
    logits.sum().backward()

    assert torch.allclose(logits, torch.tensor([[2.0, 0.0], [0.0, 2.0]]), atol=1e-6)
    assert embeddings.grad is not None
    assert head.weight.grad is not None
    assert head.temperature.grad is not None


def test_cosine_head_temperature_bounds_are_enforced() -> None:
    """Invalid temperature ranges and initial values fail during construction."""

    with pytest.raises(ValueError, match="temperature_min"):
        CosineClassifierHead(
            embedding_dim=4,
            num_classes=2,
            temperature_init=1.0,
            temperature_min=3.0,
            temperature_max=2.0,
        )

    with pytest.raises(ValueError, match="temperature_init"):
        CosineClassifierHead(
            embedding_dim=4,
            num_classes=2,
            temperature_init=5.0,
            temperature_min=1.0,
            temperature_max=2.0,
        )


@pytest.mark.parametrize(
    "embeddings, match",
    [
        (torch.randn(2, 5), "dimension"),
        (torch.randn(0, 4), "non-empty"),
        (torch.ones(2, 4, dtype=torch.int64), "floating-point"),
    ],
)
def test_heads_reject_bad_embeddings(embeddings: torch.Tensor, match: str) -> None:
    """Embedding shape, batch, and dtype checks produce clear errors."""

    head = CosineClassifierHead(
        embedding_dim=4,
        num_classes=2,
        temperature_init=1.0,
        temperature_min=0.5,
        temperature_max=2.0,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        head(embeddings)


def test_build_classifier_head_rejects_unknown_type() -> None:
    """The factory only constructs declared classifier head types."""

    assert isinstance(
        build_classifier_head(head_type="linear", embedding_dim=4, num_classes=2),
        LinearClassifierHead,
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_classifier_head(head_type="mlp", embedding_dim=4, num_classes=2)
