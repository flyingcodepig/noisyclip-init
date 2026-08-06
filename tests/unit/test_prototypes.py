"""Unit tests for single-class prototype builders."""

from __future__ import annotations

import pytest
import torch

from noisyclip.models.prototypes import (
    MeanPrototypeBuilder,
    TrimmedMeanPrototypeBuilder,
    WeightedMeanPrototypeBuilder,
    build_prototype_builder,
)


def test_mean_trimmed_and_weighted_prototypes_are_l2_normalized() -> None:
    """Prototype builders return `[C,D]` unit-norm rows."""

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 3.0, 1.0, 2.0])

    for builder in [
        MeanPrototypeBuilder(),
        TrimmedMeanPrototypeBuilder(keep_fraction=0.5),
        WeightedMeanPrototypeBuilder(),
    ]:
        prototypes = builder.fit(embeddings, targets, weights, num_classes=2)
        assert prototypes.shape == (2, 2)
        assert torch.allclose(prototypes.norm(dim=1), torch.ones(2), atol=1e-6)


def test_weighted_prototype_uses_sample_weights() -> None:
    """Weighted means move toward higher-weight samples before normalization."""

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 0], dtype=torch.int64)
    weights = torch.tensor([3.0, 1.0])

    prototype = WeightedMeanPrototypeBuilder().fit(embeddings, targets, weights, num_classes=1)

    expected = torch.nn.functional.normalize(torch.tensor([[0.75, 0.25]]), dim=1)
    assert torch.allclose(prototype, expected)


def test_prototype_builders_reject_missing_class_and_bad_targets() -> None:
    """Missing classes and out-of-range targets fail fast."""

    embeddings = torch.eye(2)

    with pytest.raises(ValueError, match="Missing samples"):
        MeanPrototypeBuilder().fit(
            embeddings,
            torch.tensor([0, 0], dtype=torch.int64),
            None,
            num_classes=2,
        )

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        MeanPrototypeBuilder().fit(
            embeddings,
            torch.tensor([0, 2], dtype=torch.int64),
            None,
            num_classes=2,
        )


def test_weighted_prototype_rejects_zero_class_weight_and_nonfinite() -> None:
    """Zero effective class weights and NaN/Inf values are invalid."""

    embeddings = torch.eye(2)
    targets = torch.tensor([0, 1], dtype=torch.int64)

    with pytest.raises(ValueError, match="zero"):
        WeightedMeanPrototypeBuilder().fit(
            embeddings,
            targets,
            torch.tensor([1.0, 0.0]),
            num_classes=2,
        )

    bad_embeddings = embeddings.clone()
    bad_embeddings[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        MeanPrototypeBuilder().fit(bad_embeddings, targets, None, num_classes=2)


def test_prototype_builder_factory_rejects_unknown_method() -> None:
    """The factory only accepts declared prototype methods."""

    assert isinstance(build_prototype_builder("mean"), MeanPrototypeBuilder)
    with pytest.raises(ValueError, match="Unsupported"):
        build_prototype_builder("median")  # type: ignore[arg-type]
