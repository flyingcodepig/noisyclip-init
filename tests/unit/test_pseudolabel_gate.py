"""Unit tests for strict pseudo-label gating."""

from __future__ import annotations

import pytest
import torch

from noisyclip.noise.pseudolabel import PseudoLabelGate
from noisyclip.noise.state import SampleState


def _state(
    sample_id: str,
    *,
    seen_count: int = 5,
    stability: float = 0.9,
    margin: float = 0.5,
) -> SampleState:
    return SampleState(
        sample_id,
        seen_count,
        0.2,
        None,
        stability,
        0.5,
        0.5,
        margin,
        0.5,
        0.5,
        "trusted",
        None,
        None,
        0,
    )


def _gate(**overrides: object) -> PseudoLabelGate:
    params = {
        "enabled": True,
        "start_epoch": 2,
        "confidence_threshold": 0.8,
        "stability_window": 3,
        "minimum_prototype_margin": 0.2,
        "maximum_dataset_fraction": 0.5,
        "pseudo_mix": 0.7,
        "stability_threshold": 0.8,
    }
    params.update(overrides)
    return PseudoLabelGate(**params)


def test_pseudolabel_default_disabled_leaves_states_unchanged() -> None:
    """The default gate is off and does not assign pseudo labels."""

    states = [_state("a")]
    logits = torch.tensor([[0.0, 5.0]])

    updated = PseudoLabelGate().apply(states, logits, epoch=999)

    assert updated == states
    assert updated[0].pseudo_target is None


def test_pseudolabel_requires_all_gates_and_start_epoch() -> None:
    """Epoch, confidence, seen-count stability, and margin gates are conjunctive."""

    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    states = [
        _state("ok"),
        _state("low_seen", seen_count=1),
        _state("low_stability", stability=0.1),
        _state("low_margin", margin=0.1),
    ]
    gate = _gate(maximum_dataset_fraction=1.0)

    before_start = gate.apply(states, logits, epoch=1)
    after_start = gate.apply(states, logits, epoch=2)

    assert all(state.pseudo_target is None for state in before_start)
    assert [state.pseudo_target for state in after_start] == [1, None, None, None]


def test_pseudolabel_confidence_and_maximum_fraction_limit() -> None:
    """Only top eligible predictions up to the dataset fraction are enabled."""

    states = [_state(str(index)) for index in range(4)]
    logits = torch.tensor(
        [
            [0.0, 5.0],
            [0.0, 4.0],
            [0.0, 3.0],
            [0.0, 0.1],
        ]
    )
    gate = _gate(maximum_dataset_fraction=0.5)

    updated = gate.apply(states, logits, epoch=2)

    assert sum(state.pseudo_target is not None for state in updated) == 2
    assert [state.pseudo_target for state in updated] == [1, 1, None, None]


def test_pseudolabel_mixed_targets_preserve_original_and_pseudo_ratio() -> None:
    """Mixed targets allocate `pseudo_mix` to pseudo labels only when present."""

    gate = _gate(pseudo_mix=0.7)
    states = [
        _state("a"),
        _state("b"),
    ]
    states[0].pseudo_target = 1

    mixed = gate.build_mixed_targets(
        torch.tensor([0, 0], dtype=torch.int64),
        states,
        num_classes=2,
    )

    assert torch.allclose(mixed[0], torch.tensor([0.3, 0.7]))
    assert torch.allclose(mixed[1], torch.tensor([1.0, 0.0]))


def test_pseudolabel_rejects_test_split_and_bad_logits() -> None:
    """Pseudo labels cannot be generated from test predictions."""

    gate = _gate()

    with pytest.raises(ValueError, match="test"):
        gate.apply([_state("a")], torch.tensor([[0.0, 5.0]]), epoch=2, split="test")

    with pytest.raises(ValueError, match="logits"):
        gate.apply([_state("a")], torch.tensor([0.0, 5.0]), epoch=2)
