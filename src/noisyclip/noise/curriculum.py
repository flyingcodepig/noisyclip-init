"""Epoch-based curriculum schedule for partition supervision weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from noisyclip.noise.state import VALID_PARTITIONS, SampleState


class _CurriculumConfigLike(Protocol):
    enabled: bool
    trusted_start_epoch: int | None
    uncertain_start_epoch: int | None
    suspicious_start_epoch: int | None
    ramp_epochs: int | None


@dataclass(frozen=True, slots=True)
class PartitionCurriculum:
    """Scale supervised weights by partition according to epoch.

    Args:
        enabled: When false, `apply` returns the input states unchanged.
        trusted_start_epoch: First epoch where trusted samples receive non-zero
            curriculum scale; `None` means epoch `0`.
        uncertain_start_epoch: First epoch where uncertain samples start.
        suspicious_start_epoch: First epoch where suspicious samples start.
        ramp_epochs: Positive ramp length from `0` to `1`.

    Returned weights always remain in `[0, 1]`. The input states are not
    mutated when the schedule is enabled, and are returned unchanged when the
    schedule is disabled.

    Raises:
        ValueError: If epochs or ramp settings are invalid.
    """

    enabled: bool = False
    trusted_start_epoch: int | None = None
    uncertain_start_epoch: int | None = None
    suspicious_start_epoch: int | None = None
    ramp_epochs: int | None = None

    @classmethod
    def from_config(cls, curriculum_config: _CurriculumConfigLike) -> PartitionCurriculum:
        """Create a curriculum from a validated config-like object."""

        return cls(
            enabled=bool(curriculum_config.enabled),
            trusted_start_epoch=curriculum_config.trusted_start_epoch,
            uncertain_start_epoch=curriculum_config.uncertain_start_epoch,
            suspicious_start_epoch=curriculum_config.suspicious_start_epoch,
            ramp_epochs=curriculum_config.ramp_epochs,
        )

    def apply(self, states: Sequence[SampleState], epoch: int) -> list[SampleState]:
        """Return epoch-adjusted states with `[0, 1]` supervised weights."""

        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        if not self.enabled:
            return list(states)
        self._validate()
        updated: list[SampleState] = []
        for state in states:
            if state.partition not in VALID_PARTITIONS:
                raise ValueError(
                    f"Invalid partition for sample_id={state.sample_id}: {state.partition!r}."
                )
            scale = self._scale_for_partition(state.partition, epoch)
            new_weight = min(1.0, max(0.0, state.supervised_weight * scale))
            updated.append(replace(state, supervised_weight=new_weight, updated_epoch=epoch))
        return updated

    def _validate(self) -> None:
        starts = (
            self.trusted_start_epoch,
            self.uncertain_start_epoch,
            self.suspicious_start_epoch,
        )
        for start in starts:
            if start is not None and start < 0:
                raise ValueError("curriculum start epochs must be non-negative.")
        if self.ramp_epochs is not None and self.ramp_epochs <= 0:
            raise ValueError("ramp_epochs must be positive when provided.")

    def _scale_for_partition(self, partition: str, epoch: int) -> float:
        start = {
            "trusted": self.trusted_start_epoch,
            "uncertain": self.uncertain_start_epoch,
            "suspicious": self.suspicious_start_epoch,
        }[partition]
        start_epoch = 0 if start is None else start
        if epoch < start_epoch:
            return 0.0
        ramp = 1 if self.ramp_epochs is None else self.ramp_epochs
        return min(1.0, (epoch - start_epoch + 1) / ramp)


def apply_curriculum(
    states: Sequence[SampleState],
    epoch: int,
    *,
    enabled: bool = False,
    trusted_start_epoch: int | None = None,
    uncertain_start_epoch: int | None = None,
    suspicious_start_epoch: int | None = None,
    ramp_epochs: int | None = None,
) -> list[SampleState]:
    """Convenience wrapper returning curriculum-adjusted `SampleState` values."""

    return PartitionCurriculum(
        enabled=enabled,
        trusted_start_epoch=trusted_start_epoch,
        uncertain_start_epoch=uncertain_start_epoch,
        suspicious_start_epoch=suspicious_start_epoch,
        ramp_epochs=ramp_epochs,
    ).apply(states, epoch)
