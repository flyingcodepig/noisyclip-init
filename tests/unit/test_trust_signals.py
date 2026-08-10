"""Unit tests for raw trust-signal computations."""

from __future__ import annotations

import pytest
import torch

from noisyclip.data.records import Batch
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.signals import (
    AugmentationAgreementSignal,
    EmaLossSignal,
    PredictionStabilitySignal,
    PrototypeMarginSignal,
    PrototypeSimilaritySignal,
    update_prediction_history,
)
from noisyclip.noise.state import SampleState


def _batch() -> Batch:
    images = torch.zeros((2, 3, 224, 224))
    return Batch(
        sample_ids=["a", "b"],
        image_weak=images,
        image_strong=images,
        targets=torch.tensor([0, 1], dtype=torch.int64),
        class_ids=["0001", "0002"],
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=logits,
        embedding=torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1),
        temperature=None,
    )


def _states() -> list[SampleState]:
    return [
        SampleState(
            "a", 2, 0.4, [0.9, 0.1], 0.9, 0.9, 0.8, 0.7, 0.8, 0.8, "trusted", None, None, 0
        ),
        SampleState(
            "b", 2, 0.6, [0.2, 0.8], 0.8, 0.8, 0.7, 0.6, 0.7, 0.7, "uncertain", None, None, 0
        ),
    ]


def test_raw_signals_return_batch_vectors() -> None:
    """All implemented signals return finite `[B]` raw values."""

    batch = _batch()
    weak = _output(torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
    strong = _output(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
    states = _states()
    prototypes = torch.eye(2)

    signals = [
        EmaLossSignal(momentum=0.5),
        AugmentationAgreementSignal(),
        PrototypeSimilaritySignal(),
        PrototypeMarginSignal(),
        PredictionStabilitySignal(),
    ]

    for signal in signals:
        values = signal.compute(batch, weak, strong, states, prototypes)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()


def test_ema_loss_uses_previous_state_and_targets() -> None:
    """EMA loss blends previous state with current cross-entropy."""

    values = EmaLossSignal(momentum=0.5).compute(
        _batch(),
        _output(torch.tensor([[2.0, 0.0], [0.0, 2.0]])),
        None,
        _states(),
        None,
    )

    current = torch.nn.functional.cross_entropy(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([0, 1], dtype=torch.int64),
        reduction="none",
    )
    assert torch.allclose(values, 0.5 * torch.tensor([0.4, 0.6]) + 0.5 * current)


def test_missing_strong_view_and_prototypes_fail_clearly() -> None:
    """Signals with required dependencies reject absent inputs."""

    batch = _batch()
    output = _output(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="output_strong"):
        AugmentationAgreementSignal().compute(batch, output, None, _states(), None)

    with pytest.raises(ValueError, match="prototypes"):
        PrototypeSimilaritySignal().compute(batch, output, None, _states(), None)


def test_state_and_batch_sample_id_mismatch_fails() -> None:
    """Signals require state entries to align with `batch.sample_ids`."""

    batch = _batch()
    output = _output(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    states = list(reversed(_states()))

    with pytest.raises(ValueError, match="state order"):
        PredictionStabilitySignal().compute(batch, output, None, states, None)


def test_prediction_history_updates_probabilities_and_seen_count() -> None:
    """Epoch history stores normalized detached EMA probabilities by sample ID."""

    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
    updated = update_prediction_history(_states(), logits, epoch=1, momentum=0.5)

    assert [state.sample_id for state in updated] == ["a", "b"]
    assert [state.seen_count for state in updated] == [3, 3]
    assert all(state.updated_epoch == 1 for state in updated)
    assert all(state.ema_probs is not None for state in updated)
    assert all(sum(state.ema_probs or []) == pytest.approx(1.0) for state in updated)
    assert [state.prediction_history for state in updated] == [[0], [1]]
    assert logits.grad is None


def test_prediction_stability_uses_a_fixed_top1_window() -> None:
    """Stability is neutral until the configured history window is full."""

    batch = _batch()
    states = _states()
    states[0].prediction_history = [0, 0]
    states[1].prediction_history = [0, 1]
    output = _output(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))

    values = PredictionStabilitySignal(window=3).compute(batch, output, None, states, None)

    assert values.tolist() == pytest.approx([1.0, 2.0 / 3.0])


def test_prediction_history_rejects_wrong_probability_history() -> None:
    """Stored probability vectors must match class count and sum to one."""

    states = _states()
    states[0].ema_probs = [0.2, 0.2]

    with pytest.raises(ValueError, match="sum to 1"):
        update_prediction_history(states, torch.zeros((2, 2)), epoch=1)
