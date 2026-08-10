"""Unit tests for deterministic class-wise trust partitioning."""

from __future__ import annotations

import pytest
import torch

from noisyclip.noise.partition import (
    apply_partitions,
    apply_supervision_weights,
    partition_by_class,
)
from noisyclip.noise.state import SampleState


def _state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id, 1, 0.2, None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, "uncertain", None, None, 0
    )


def test_partition_is_mutually_exclusive_and_covers_all_samples() -> None:
    """Every sample receives exactly one class-wise quantile partition."""

    sample_ids = ["a", "b", "c", "d", "e", "f"]
    targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    scores = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.6, 1.0])

    partitions = partition_by_class(
        sample_ids,
        targets,
        scores,
        trusted_quantile=0.65,
        uncertain_quantile=0.90,
        min_samples_per_class=2,
    )

    assert set(partitions) == set(sample_ids)
    assert set(partitions.values()) <= {"trusted", "uncertain", "suspicious"}
    assert partitions["c"] == "trusted"
    assert partitions["f"] == "trusted"
    assert partitions["a"] == "suspicious"
    assert partitions["d"] == "suspicious"


def test_small_class_strategy_is_deterministic_uncertain() -> None:
    """Classes smaller than the minimum are marked uncertain."""

    partitions = partition_by_class(
        ["solo"],
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0]),
        trusted_quantile=0.65,
        uncertain_quantile=0.90,
        min_samples_per_class=2,
    )

    assert partitions == {"solo": "uncertain"}


def test_partition_rejects_duplicate_ids_and_bad_ranges() -> None:
    """Invalid IDs and trust-score ranges fail fast."""

    with pytest.raises(ValueError, match="duplicate"):
        partition_by_class(
            ["a", "a"],
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([0.1, 0.2]),
            trusted_quantile=0.65,
            uncertain_quantile=0.90,
        )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        partition_by_class(
            ["a"],
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1.5]),
            trusted_quantile=0.65,
            uncertain_quantile=0.90,
        )


def test_apply_partitions_updates_states_and_rejects_invalid_partition() -> None:
    """Partition application validates coverage and partition names."""

    states = [_state("a"), _state("b")]
    updated = apply_partitions(states, {"a": "trusted", "b": "suspicious"}, epoch=1)

    assert [state.partition for state in updated] == ["trusted", "suspicious"]
    assert [state.updated_epoch for state in updated] == [1, 1]

    with pytest.raises(ValueError, match="Invalid partition"):
        apply_partitions(states, {"a": "clean", "b": "trusted"}, epoch=1)


def test_supervision_weight_mapping_uses_all_configured_bounds() -> None:
    """Partition weights are rebuilt from trust instead of reusing a scaled weight."""

    states = [_state("a"), _state("b"), _state("c")]
    states[0].partition = "trusted"
    states[0].supervised_weight = 0.01
    states[1].partition = "uncertain"
    states[1].trust_score = 0.5
    states[1].supervised_weight = 0.01
    states[2].partition = "suspicious"
    states[2].supervised_weight = 0.01

    updated = apply_supervision_weights(
        states,
        trusted=1.0,
        uncertain_min=0.3,
        uncertain_max=0.7,
        suspicious=0.1,
        epoch=2,
    )

    assert [state.supervised_weight for state in updated] == pytest.approx([1.0, 0.5, 0.1])
    assert [state.updated_epoch for state in updated] == [2, 2, 2]
