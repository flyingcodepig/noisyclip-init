"""Loss result records and loss protocols from the shared contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


@dataclass(slots=True)
class LossOutput:
    """Composite loss output consumed by the training engine.

    `total` and every tensor in `components` must be scalar finite tensors.
    `per_sample_supervised`, when present, has shape `[B]` and must be detached
    before state stores consume it.
    """

    total: Tensor
    components: Mapping[str, Tensor]
    per_sample_supervised: Tensor | None


class LossTerm(Protocol):
    """Protocol for one weighted loss component."""

    name: str

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return a scalar loss, optionally with a `[B]` per-sample tensor."""


class CompositeLoss(Protocol):
    """Protocol for configured loss composition."""

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> LossOutput:
        """Return total, component scalars, and optional per-sample loss."""
