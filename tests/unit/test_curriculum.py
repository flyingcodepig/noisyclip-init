"""Unit tests for partition curriculum scheduling."""

from __future__ import annotations

import pytest

from noisyclip.noise.curriculum import PartitionCurriculum, apply_curriculum
from noisyclip.noise.state import SampleState


def _state(sample_id: str, partition: str, weight: float) -> SampleState:
    return SampleState(
        sample_id, 1, 0.2, None, 0.5, 0.5, 0.5, 0.5, 0.5, weight, partition, None, None, 0
    )


def test_disabled_curriculum_does_not_change_input_state() -> None:
    """Default-disabled curriculum returns the same state values."""

    states = [_state("a", "trusted", 0.8)]

    updated = apply_curriculum(states, epoch=5)

    assert updated == states
    assert updated[0].supervised_weight == 0.8
    assert states[0].updated_epoch == 0


def test_enabled_curriculum_scales_weights_by_epoch_and_partition() -> None:
    """Enabled schedule gates partitions and ramps weights into `[0,1]`."""

    states = [
        _state("a", "trusted", 1.0),
        _state("b", "uncertain", 0.6),
        _state("c", "suspicious", 0.5),
    ]
    curriculum = PartitionCurriculum(
        enabled=True,
        trusted_start_epoch=0,
        uncertain_start_epoch=2,
        suspicious_start_epoch=4,
        ramp_epochs=2,
    )

    updated = curriculum.apply(states, epoch=2)

    assert [state.supervised_weight for state in updated] == [1.0, 0.3, 0.0]
    assert all(0.0 <= state.supervised_weight <= 1.0 for state in updated)
    assert states[1].supervised_weight == 0.6


def test_curriculum_rejects_invalid_epoch_and_partition() -> None:
    """Invalid epochs and partitions produce explicit errors."""

    curriculum = PartitionCurriculum(enabled=True)

    with pytest.raises(ValueError, match="epoch"):
        curriculum.apply([_state("a", "trusted", 1.0)], epoch=-1)

    with pytest.raises(ValueError, match="Invalid partition"):
        curriculum.apply([_state("a", "clean", 1.0)], epoch=0)
