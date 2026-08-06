"""Early-learning regularization keyed by stable sample IDs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.losses._validation import (
    require_batch_alignment,
    require_model_output,
    require_scalar,
)
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


class ELRLoss:
    """Early-learning regularization with sample-ID historical targets.

    Args:
        target_momentum: Momentum in `[0, 1)` for historical probability targets.
        start_epoch: First epoch at which the regularizer updates history and
            contributes a scalar value.
        enabled: When false, no inputs beyond logits shape are required and a
            finite zero scalar is returned.

    Inputs:
        Active calls use finite logits `[B, C]`, unique `batch.sample_ids`, and
        batch-aligned sample states. Historical targets are stored by
        `sample_id`, not batch position.

    Outputs:
        A scalar ELR term. The probability target branch and stored history are
        detached tensors, so gradients only flow through current student logits.

    Raises:
        ValueError: If IDs are duplicated, tensors are malformed/non-finite, or a
            loaded history entry is incompatible with the current class count.
    """

    name = "loss/elr"

    def __init__(
        self,
        *,
        target_momentum: float = 0.7,
        start_epoch: int = 0,
        enabled: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        if not 0.0 <= target_momentum < 1.0:
            raise ValueError("target_momentum must be in the range [0, 1).")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.target_momentum = target_momentum
        self.start_epoch = start_epoch
        self.enabled = enabled
        self.epsilon = epsilon
        self._targets: dict[str, Tensor] = {}
        self._num_classes: int | None = None

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        sample_states: list[SampleState],
        epoch: int,
    ) -> Tensor:
        """Compute the active ELR scalar for logits `[B, C]`.

        Args:
            batch: Batch with unique sample IDs.
            student_weak: Student output with finite logits `[B, C]`.
            sample_states: Batch-aligned states used for ID consistency checks.
            epoch: Non-negative epoch index controlling warmup.

        Returns:
            A finite scalar tensor. Returns zero when disabled or before
            `start_epoch`.

        Raises:
            ValueError: If epoch is negative, IDs are invalid, or history has an
                incompatible shape.
        """

        batch_size, num_classes = require_model_output("student_weak", student_weak)
        require_batch_alignment(batch, batch_size, sample_states)
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        if not self.enabled or epoch < self.start_epoch:
            zero = student_weak.logits.new_zeros(())
            require_scalar(self.name, zero)
            return zero

        if self._num_classes is None:
            self._num_classes = num_classes
        elif self._num_classes != num_classes:
            raise ValueError(
                f"ELR history was initialized for {self._num_classes} classes, got {num_classes}."
            )

        probabilities = F.softmax(student_weak.logits, dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("ELR probabilities must be finite.")
        detached_probabilities = probabilities.detach()

        history_rows: list[Tensor] = []
        for row, sample_id in zip(detached_probabilities, batch.sample_ids, strict=True):
            previous = self._targets.get(sample_id)
            if previous is None:
                updated = row.detach().cpu()
            else:
                self._validate_history_row(sample_id, previous, num_classes)
                updated = (
                    self.target_momentum * previous
                    + (1.0 - self.target_momentum) * row.detach().cpu()
                )
            self._targets[sample_id] = updated.detach().clone()
            history_rows.append(updated.to(device=probabilities.device, dtype=probabilities.dtype))

        target_history = torch.stack(history_rows, dim=0).detach()
        agreement = (probabilities * target_history).sum(dim=1)
        loss = torch.log(torch.clamp(1.0 - agreement, min=self.epsilon)).mean()
        require_scalar(self.name, loss)
        return loss

    def state_dict(self) -> dict[str, object]:
        """Return checkpointable detached CPU history.

        Returns:
            A dictionary containing `target_momentum`, `start_epoch`,
            `enabled`, `epsilon`, `num_classes`, and a mapping of sample ID to
            probability vector `[C]`.
        """

        return {
            "target_momentum": self.target_momentum,
            "start_epoch": self.start_epoch,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "num_classes": self._num_classes,
            "targets": {key: value.detach().cpu().clone() for key, value in self._targets.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore checkpointed detached ELR history.

        Args:
            state_dict: Mapping produced by `state_dict`.

        Raises:
            TypeError: If fields have incompatible types.
            ValueError: If history vectors are non-finite, non-1D, or malformed.
        """

        num_classes_raw = state_dict.get("num_classes")
        if num_classes_raw is not None and not isinstance(num_classes_raw, int):
            raise TypeError("ELR state num_classes must be an int or None.")
        targets_raw = state_dict.get("targets", {})
        if not isinstance(targets_raw, Mapping):
            raise TypeError("ELR state targets must be a mapping.")

        restored: dict[str, Tensor] = {}
        for sample_id, value in targets_raw.items():
            if not isinstance(sample_id, str):
                raise TypeError("ELR state target keys must be sample_id strings.")
            if not isinstance(value, Tensor):
                raise TypeError(f"ELR history for {sample_id} must be a tensor.")
            tensor = value.detach().cpu().to(dtype=torch.float32)
            if tensor.ndim != 1:
                raise ValueError(f"ELR history for {sample_id} must have shape [C].")
            if num_classes_raw is not None and tensor.shape[0] != num_classes_raw:
                raise ValueError(
                    f"ELR history for {sample_id} has {tensor.shape[0]} classes, "
                    f"expected {num_classes_raw}."
                )
            self._validate_history_row(sample_id, tensor, int(tensor.shape[0]))
            restored[sample_id] = tensor.clone()

        self._num_classes = num_classes_raw
        self._targets = restored
        self._load_optional_hyperparameters(state_dict)

    def _validate_history_row(self, sample_id: str, value: Tensor, num_classes: int) -> None:
        if value.shape != (num_classes,):
            raise ValueError(
                f"ELR history for {sample_id} must have shape [{num_classes}], "
                f"got {tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"ELR history for {sample_id} must be finite.")
        if bool((value < 0).any() or (value > 1).any()):
            raise ValueError(f"ELR history for {sample_id} must be in [0, 1].")

    def _load_optional_hyperparameters(self, state_dict: Mapping[str, object]) -> None:
        target_momentum = state_dict.get("target_momentum", self.target_momentum)
        start_epoch = state_dict.get("start_epoch", self.start_epoch)
        enabled = state_dict.get("enabled", self.enabled)
        epsilon = state_dict.get("epsilon", self.epsilon)
        if not isinstance(target_momentum, float | int):
            raise TypeError("ELR state target_momentum must be numeric.")
        if not isinstance(start_epoch, int):
            raise TypeError("ELR state start_epoch must be an int.")
        if not isinstance(enabled, bool):
            raise TypeError("ELR state enabled must be bool.")
        if not isinstance(epsilon, float | int):
            raise TypeError("ELR state epsilon must be numeric.")
        self.target_momentum = float(target_momentum)
        self.start_epoch = start_epoch
        self.enabled = enabled
        self.epsilon = float(epsilon)
