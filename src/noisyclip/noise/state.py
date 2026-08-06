"""Sample noise-state records and trust/prototype protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from torch import Tensor

from noisyclip.data.records import Batch, SampleRecord
from noisyclip.models.outputs import ModelOutput


@dataclass(slots=True)
class SampleState:
    """Persistent per-sample trust state for noise-aware training.

    Scores and weights use the closed range `[0, 1]`. `partition` is one of
    `trusted`, `uncertain`, or `suspicious`. `pseudo_target` is an internal
    class index or `None` when no pseudo-label is active.
    """

    sample_id: str
    seen_count: int
    ema_loss: float
    ema_probs: list[float] | None
    prediction_stability: float
    augmentation_agreement: float
    prototype_similarity: float
    prototype_margin: float
    trust_score: float
    supervised_weight: float
    partition: str
    pseudo_target: int | None
    pseudo_confidence: float | None
    updated_epoch: int


class PrototypeBuilder(Protocol):
    """Protocol for building class prototypes from embeddings."""

    def fit(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """Return `[C, D]` L2-normalized prototypes or raise on missing classes."""


class TrustSignal(Protocol):
    """Protocol for computing one raw per-sample trust signal."""

    name: str

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` signal values without normalization or persistence."""


class TrustAggregator(Protocol):
    """Protocol for class-aware aggregation of trust signals."""

    def update_epoch(
        self,
        records: list[SampleRecord],
        raw_signals: Mapping[str, Tensor],
        previous: list[SampleState],
        epoch: int,
    ) -> list[SampleState]:
        """Return validated epoch states after class-wise normalization."""


class SampleStateStore(Protocol):
    """Protocol for transactional storage of per-sample state."""

    def load(self, sample_ids: list[str]) -> list[SampleState]:
        """Load states matching `sample_ids` in the requested order."""

    def stage_epoch(self, states: list[SampleState], epoch: int) -> Path:
        """Write uncommitted epoch state and return the staged path."""

    def commit_epoch(self, epoch: int) -> None:
        """Atomically publish the staged state for `epoch`."""

    def rollback_uncommitted(self) -> None:
        """Remove staged, uncommitted state without touching committed epochs."""
