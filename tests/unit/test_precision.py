"""Precision-manager tests for safe accumulation boundaries."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from noisyclip.engine.precision import PrecisionConfig, PrecisionManager


def test_partial_accumulation_group_is_flushed_as_an_average() -> None:
    """A final short group must update once without shrinking its gradient."""

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    precision = PrecisionManager(
        PrecisionConfig(gradient_accumulation_steps=2, gradient_clip_norm=100.0),
        device=torch.device("cpu"),
    )
    optimizer.zero_grad(set_to_none=True)
    steps = 0
    for index in range(3):
        precision.backward(model(torch.ones(1, 1)).sum())
        steps += int(
            precision.step_if_needed(
                microbatch_index=index,
                model=model,
                optimizer=optimizer,
            )
        )
    steps += int(
        precision.step_if_needed(
            microbatch_index=3,
            model=model,
            optimizer=optimizer,
            force=True,
        )
    )

    assert steps == 2
    assert torch.allclose(model.weight, torch.tensor([[-1.0]]))


def test_gradient_validator_runs_before_gradients_are_cleared() -> None:
    """Frozen-parameter guards must observe gradients before optimizer zeroing."""

    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    precision = PrecisionManager(PrecisionConfig(), device=torch.device("cpu"))
    precision.backward(model(torch.ones(1, 1)).sum())
    observed: list[torch.Tensor] = []

    precision.step_if_needed(
        microbatch_index=0,
        model=model,
        optimizer=optimizer,
        gradient_validator=lambda: observed.append(model.weight.grad.detach().clone()),
    )

    assert len(observed) == 1
    assert torch.equal(observed[0], torch.ones(1, 1))
    assert model.weight.grad is None


def test_checkpoint_state_refuses_pending_gradients() -> None:
    """A checkpoint cannot silently omit a partially accumulated step."""

    model = nn.Linear(1, 1, bias=False)
    precision = PrecisionManager(
        PrecisionConfig(gradient_accumulation_steps=2), device=torch.device("cpu")
    )
    precision.backward(model(torch.ones(1, 1)).sum())
    with pytest.raises(RuntimeError, match="unstepped"):
        precision.state_dict()
