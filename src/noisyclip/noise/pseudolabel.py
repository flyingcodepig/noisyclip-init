"""Strict pseudo-label gating for train/validation samples only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.noise.state import SampleState


class _PseudoLabelConfigLike(Protocol):
    enabled: bool
    start_epoch: int
    confidence_threshold: float
    stability_window: int
    minimum_prototype_margin: float
    maximum_dataset_fraction: float
    pseudo_mix: float


@dataclass(frozen=True, slots=True)
class PseudoLabelGate:
    """Apply conservative pseudo-label eligibility gates.

    Args:
        enabled: When false, `apply` returns states unchanged.
        start_epoch: First epoch where pseudo-labeling may activate.
        confidence_threshold: Minimum max softmax probability in `(0, 1]`.
        stability_window: Minimum `seen_count` required for stability evidence.
        minimum_prototype_margin: Minimum stored normalized prototype margin.
        maximum_dataset_fraction: Maximum fraction of the provided state set
            allowed to receive pseudo labels.
        pseudo_mix: Ratio assigned to the pseudo target when building mixed
            targets; the original label keeps `1 - pseudo_mix`.
        stability_threshold: Minimum stored prediction stability in `[0, 1]`.

    `logits` must be shaped `[N, C]`. The returned states preserve order and
    set `pseudo_target` plus `pseudo_confidence` only for eligible samples.

    Raises:
        ValueError: If called for `split="test"`, gate ranges are invalid,
            logits/states are malformed, or outputs would be non-finite.
    """

    enabled: bool = False
    start_epoch: int = 999
    confidence_threshold: float = 0.98
    stability_window: int = 5
    minimum_prototype_margin: float = 0.20
    maximum_dataset_fraction: float = 0.05
    pseudo_mix: float = 0.8
    stability_threshold: float = 0.8

    @classmethod
    def from_config(cls, pseudolabel_config: _PseudoLabelConfigLike) -> PseudoLabelGate:
        """Create a pseudo-label gate from a validated config-like object."""

        return cls(
            enabled=bool(pseudolabel_config.enabled),
            start_epoch=int(pseudolabel_config.start_epoch),
            confidence_threshold=float(pseudolabel_config.confidence_threshold),
            stability_window=int(pseudolabel_config.stability_window),
            minimum_prototype_margin=float(pseudolabel_config.minimum_prototype_margin),
            maximum_dataset_fraction=float(pseudolabel_config.maximum_dataset_fraction),
            pseudo_mix=float(pseudolabel_config.pseudo_mix),
        )

    def apply(
        self,
        states: Sequence[SampleState],
        logits: Tensor,
        *,
        epoch: int,
        split: str = "train",
    ) -> list[SampleState]:
        """Return states after applying pseudo-label gates to `[N, C]` logits."""

        if not self.enabled:
            return list(states)
        self._validate(split=split, epoch=epoch)
        _validate_logits(logits, len(states))
        if epoch < self.start_epoch:
            return [
                replace(state, pseudo_target=None, pseudo_confidence=None, updated_epoch=epoch)
                for state in states
            ]
        probs = logits.softmax(dim=1)
        confidences, targets = probs.max(dim=1)
        eligible_indices = [
            index
            for index, state in enumerate(states)
            if self._is_state_eligible(state, confidence=float(confidences[index].item()))
        ]
        max_count = math.floor(len(states) * self.maximum_dataset_fraction)
        chosen = sorted(
            eligible_indices,
            key=lambda index: (-float(confidences[index].item()), states[index].sample_id),
        )[:max_count]
        chosen_set = set(chosen)
        updated: list[SampleState] = []
        for index, state in enumerate(states):
            if index in chosen_set:
                updated.append(
                    replace(
                        state,
                        pseudo_target=int(targets[index].item()),
                        pseudo_confidence=float(confidences[index].item()),
                        updated_epoch=epoch,
                    )
                )
            else:
                updated.append(
                    replace(state, pseudo_target=None, pseudo_confidence=None, updated_epoch=epoch)
                )
        return updated

    def build_mixed_targets(
        self,
        original_targets: Tensor,
        states: Sequence[SampleState],
        *,
        num_classes: int,
    ) -> Tensor:
        """Build `[N, C]` soft targets blending original and pseudo labels.

        Args:
            original_targets: Int64 tensor shaped `[N]` with class ids in
                `[0, C)`.
            states: States aligned to `original_targets`; pseudo labels are
                used only when `pseudo_target` is not `None`.
            num_classes: Positive class count `C`.

        Returns:
            Floating-point `[N, C]` target distributions. Rows sum to `1`.

        Raises:
            TypeError: If target dtype is invalid.
            ValueError: If shapes/ranges are invalid or pseudo targets are out
                of range.
        """

        _validate_targets(original_targets, len(states), num_classes)
        mixed = F.one_hot(original_targets, num_classes=num_classes).to(torch.float32)
        for index, state in enumerate(states):
            if state.pseudo_target is None:
                continue
            if not 0 <= state.pseudo_target < num_classes:
                raise ValueError(
                    f"pseudo_target out of range for sample_id={state.sample_id}: "
                    f"{state.pseudo_target}."
                )
            pseudo = torch.zeros(num_classes, dtype=torch.float32, device=mixed.device)
            pseudo[state.pseudo_target] = 1.0
            mixed[index] = (1.0 - self.pseudo_mix) * mixed[index] + self.pseudo_mix * pseudo
        return mixed

    def _validate(self, *, split: str, epoch: int) -> None:
        if split == "test":
            raise ValueError("Pseudo-labels must not be generated from test predictions.")
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val for pseudo-labeling, got {split!r}.")
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative.")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1].")
        if self.stability_window < 1:
            raise ValueError("stability_window must be positive.")
        if not 0.0 < self.maximum_dataset_fraction <= 1.0:
            raise ValueError("maximum_dataset_fraction must be in (0, 1].")
        if not 0.0 < self.pseudo_mix < 1.0:
            raise ValueError("pseudo_mix must preserve original and pseudo targets.")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be in [0, 1].")

    def _is_state_eligible(self, state: SampleState, *, confidence: float) -> bool:
        return (
            confidence >= self.confidence_threshold
            and state.seen_count >= self.stability_window
            and state.prediction_stability >= self.stability_threshold
            and state.prototype_margin >= self.minimum_prototype_margin
        )


def _validate_logits(logits: Tensor, expected_count: int) -> None:
    if logits.ndim != 2 or logits.shape[0] != expected_count or logits.shape[1] <= 0:
        raise ValueError(f"logits must have shape [N, C], got {tuple(logits.shape)}.")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains NaN or Inf values.")


def _validate_targets(targets: Tensor, expected_count: int, num_classes: int) -> None:
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    if targets.ndim != 1 or targets.shape[0] != expected_count:
        raise ValueError(f"original_targets must have shape [N], got {tuple(targets.shape)}.")
    if targets.dtype != torch.int64:
        raise TypeError("original_targets must be an int64 tensor.")
    if bool((targets < 0).any()) or bool((targets >= num_classes).any()):
        raise ValueError(f"original_targets must be in [0, {num_classes}).")
