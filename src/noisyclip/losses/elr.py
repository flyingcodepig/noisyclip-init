"""Early-learning regularization keyed by stable sample IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.losses._validation import (
    require_batch_alignment,
    require_model_output,
    require_scalar,
    supervised_weights,
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
        sample_ids: Optional complete training ID order used to preallocate one
            contiguous history table instead of transferring history per batch.
        history_device: Device for the preallocated table. Assembly places it
            beside the student logits to avoid CPU-GPU synchronization.

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
        sample_ids: Sequence[str] | None = None,
        history_device: torch.device | str | None = None,
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
        configured_ids = list(sample_ids) if sample_ids is not None else None
        if configured_ids is not None:
            if any(not isinstance(sample_id, str) or not sample_id for sample_id in configured_ids):
                raise ValueError("ELR sample_ids must contain non-empty strings.")
            if len(set(configured_ids)) != len(configured_ids):
                raise ValueError("ELR sample_ids must be unique.")
        self._configured_sample_ids = configured_ids
        self._sample_to_index = (
            None
            if configured_ids is None
            else {sample_id: index for index, sample_id in enumerate(configured_ids)}
        )
        self._history_device = (
            None if history_device is None else torch.device(history_device)
        )
        self._history_tensor: Tensor | None = None
        self._history_initialized: Tensor | None = None

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
        weights = supervised_weights(
            sample_states,
            device=probabilities.device,
            dtype=probabilities.dtype,
            require_positive=False,
        )

        if self._sample_to_index is not None:
            target_history = self._preallocated_target_history(
                batch.sample_ids,
                detached_probabilities,
                weights,
                num_classes,
            )
        else:
            target_history = self._mapping_target_history(
                batch.sample_ids,
                sample_states,
                detached_probabilities,
                weights,
                num_classes,
            )
        agreement = (probabilities * target_history).sum(dim=1)
        per_sample = torch.log(torch.clamp(1.0 - agreement, min=self.epsilon))
        denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        loss = (per_sample * weights).sum() / denominator
        require_scalar(self.name, loss)
        return loss

    def _preallocated_target_history(
        self,
        sample_ids: list[str],
        probabilities: Tensor,
        weights: Tensor,
        num_classes: int,
    ) -> Tensor:
        if self._sample_to_index is None or self._configured_sample_ids is None:
            raise RuntimeError("Preallocated ELR history requires configured sample IDs.")
        missing = [sample_id for sample_id in sample_ids if sample_id not in self._sample_to_index]
        if missing:
            raise ValueError(f"ELR batch contains unknown sample_id(s): {missing}.")
        device = self._history_device or probabilities.device
        if self._history_tensor is None:
            self._history_tensor = torch.zeros(
                (len(self._configured_sample_ids), num_classes),
                device=device,
                dtype=torch.float32,
            )
            self._history_initialized = torch.zeros(
                len(self._configured_sample_ids),
                device=device,
                dtype=torch.bool,
            )
        if self._history_initialized is None:
            raise RuntimeError("ELR history initialization mask is missing.")
        if self._history_tensor.shape != (len(self._configured_sample_ids), num_classes):
            raise ValueError("Preallocated ELR history shape does not match current classes.")

        indices = torch.tensor(
            [self._sample_to_index[sample_id] for sample_id in sample_ids],
            device=device,
            dtype=torch.int64,
        )
        current = probabilities.to(device=device, dtype=torch.float32)
        previous = self._history_tensor.index_select(0, indices)
        initialized = self._history_initialized.index_select(0, indices)
        update_rate = (1.0 - self.target_momentum) * weights.to(
            device=device, dtype=torch.float32
        )
        updated = torch.where(
            initialized[:, None],
            (1.0 - update_rate[:, None]) * previous + update_rate[:, None] * current,
            current,
        )
        persistent = weights.to(device=device) > 0
        persistent_indices = indices[persistent]
        self._history_tensor.index_copy_(0, persistent_indices, updated[persistent])
        self._history_initialized[persistent_indices] = True
        return updated.to(device=probabilities.device, dtype=probabilities.dtype).detach()

    def _mapping_target_history(
        self,
        sample_ids: list[str],
        sample_states: list[SampleState],
        probabilities: Tensor,
        weights: Tensor,
        num_classes: int,
    ) -> Tensor:
        # Compatibility path for independently constructed loss objects. It
        # batches transfers; assembled training uses the GPU table above.
        target_history = probabilities.clone()
        existing_positions: list[int] = []
        existing_rows: list[Tensor] = []
        for position, sample_id in enumerate(sample_ids):
            previous = self._targets.get(sample_id)
            if previous is None:
                continue
            self._validate_history_row(sample_id, previous, num_classes)
            existing_positions.append(position)
            existing_rows.append(previous)

        if existing_positions:
            position_index = torch.tensor(
                existing_positions,
                device=probabilities.device,
                dtype=torch.int64,
            )
            previous_history = torch.stack(existing_rows).to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            update_rate = (1.0 - self.target_momentum) * weights.index_select(
                0, position_index
            )
            current = probabilities.index_select(0, position_index)
            updated = (
                (1.0 - update_rate[:, None]) * previous_history
                + update_rate[:, None] * current
            )
            target_history.index_copy_(0, position_index, updated)

        persistent_positions = [
            position
            for position, state in enumerate(sample_states)
            if state.supervised_weight > 0.0
        ]
        if persistent_positions:
            persistent_index = torch.tensor(
                persistent_positions,
                device=probabilities.device,
                dtype=torch.int64,
            )
            persisted = (
                target_history.index_select(0, persistent_index).detach().cpu().float()
            )
            for row, position in zip(persisted, persistent_positions, strict=True):
                self._targets[sample_ids[position]] = row
        return target_history.detach()

    def state_dict(self) -> dict[str, object]:
        """Return checkpointable detached CPU history.

        Returns:
            A dictionary containing the hyperparameters, ordered sample IDs,
            and one contiguous `[N, C]` probability-history tensor.
        """

        sample_ids, target_tensor = self._compact_history()
        return {
            "format_version": 2,
            "target_momentum": self.target_momentum,
            "start_epoch": self.start_epoch,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "num_classes": self._num_classes,
            "sample_ids": sample_ids,
            "target_tensor": target_tensor.detach().cpu().float().contiguous(),
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
        if num_classes_raw is not None and (
            isinstance(num_classes_raw, bool) or not isinstance(num_classes_raw, int)
        ):
            raise TypeError("ELR state num_classes must be an int or None.")
        if isinstance(num_classes_raw, int) and num_classes_raw <= 0:
            raise ValueError("ELR state num_classes must be positive when present.")
        if "sample_ids" in state_dict or "target_tensor" in state_dict:
            if state_dict.get("format_version") != 2:
                raise ValueError("Unsupported compact ELR state format_version.")
            restored = self._load_compact_targets(state_dict, num_classes_raw)
        else:
            restored = self._load_legacy_targets(state_dict, num_classes_raw)

        self._num_classes = num_classes_raw
        if self._sample_to_index is None:
            self._targets = restored
        else:
            self._restore_preallocated_history(restored, num_classes_raw)
        self._load_optional_hyperparameters(state_dict)

    def _compact_history(self) -> tuple[list[str], Tensor]:
        if self._sample_to_index is None:
            sample_ids = sorted(self._targets)
            target_tensor = (
                torch.stack([self._targets[sample_id] for sample_id in sample_ids])
                if sample_ids
                else torch.empty((0, self._num_classes or 0), dtype=torch.float32)
            )
            return sample_ids, target_tensor
        if self._configured_sample_ids is None:
            raise RuntimeError("Configured ELR sample IDs are missing.")
        if self._history_tensor is None or self._history_initialized is None:
            return [], torch.empty((0, self._num_classes or 0), dtype=torch.float32)
        initialized = self._history_initialized.detach().cpu().tolist()
        sample_ids = [
            sample_id
            for sample_id, present in zip(
                self._configured_sample_ids, initialized, strict=True
            )
            if present
        ]
        target_tensor = self._history_tensor[self._history_initialized].detach().cpu().float()
        return sample_ids, target_tensor

    def _restore_preallocated_history(
        self,
        restored: Mapping[str, Tensor],
        num_classes: int | None,
    ) -> None:
        if self._sample_to_index is None or self._configured_sample_ids is None:
            raise RuntimeError("Preallocated ELR restore requires configured sample IDs.")
        unknown = sorted(set(restored) - set(self._sample_to_index))
        if unknown:
            raise ValueError(f"ELR checkpoint contains unknown sample_id(s): {unknown}.")
        if num_classes is None:
            if restored:
                raise ValueError("ELR checkpoint history requires num_classes.")
            return
        device = self._history_device or torch.device("cpu")
        self._history_tensor = torch.zeros(
            (len(self._configured_sample_ids), num_classes),
            device=device,
            dtype=torch.float32,
        )
        self._history_initialized = torch.zeros(
            len(self._configured_sample_ids),
            device=device,
            dtype=torch.bool,
        )
        if not restored:
            return
        sample_ids = list(restored)
        indices = torch.tensor(
            [self._sample_to_index[sample_id] for sample_id in sample_ids],
            device=device,
            dtype=torch.int64,
        )
        values = torch.stack([restored[sample_id] for sample_id in sample_ids]).to(device)
        self._history_tensor.index_copy_(0, indices, values)
        self._history_initialized[indices] = True

    def _load_compact_targets(
        self,
        state_dict: Mapping[str, object],
        num_classes: int | None,
    ) -> dict[str, Tensor]:
        sample_ids_raw = state_dict.get("sample_ids")
        target_tensor_raw = state_dict.get("target_tensor")
        if not isinstance(sample_ids_raw, list) or any(
            not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids_raw
        ):
            raise TypeError("Compact ELR sample_ids must be a list of non-empty strings.")
        if len(set(sample_ids_raw)) != len(sample_ids_raw):
            raise ValueError("Compact ELR sample_ids must be unique.")
        if not isinstance(target_tensor_raw, Tensor):
            raise TypeError("Compact ELR target_tensor must be a tensor.")
        tensor = target_tensor_raw.detach().cpu().to(dtype=torch.float32).contiguous()
        expected_classes = num_classes or 0
        if tensor.shape != (len(sample_ids_raw), expected_classes):
            raise ValueError(
                "Compact ELR target_tensor must have shape "
                f"[{len(sample_ids_raw)}, {expected_classes}], got {tuple(tensor.shape)}."
            )
        restored: dict[str, Tensor] = {}
        for sample_id, row in zip(sample_ids_raw, tensor, strict=True):
            self._validate_history_row(sample_id, row, expected_classes)
            restored[sample_id] = row
        return restored

    def _load_legacy_targets(
        self,
        state_dict: Mapping[str, object],
        num_classes: int | None,
    ) -> dict[str, Tensor]:
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
            if num_classes is not None and tensor.shape[0] != num_classes:
                raise ValueError(
                    f"ELR history for {sample_id} has {tensor.shape[0]} classes, "
                    f"expected {num_classes}."
                )
            self._validate_history_row(sample_id, tensor, int(tensor.shape[0]))
            restored[sample_id] = tensor.clone()
        return restored

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
        if isinstance(target_momentum, bool) or not isinstance(target_momentum, float | int):
            raise TypeError("ELR state target_momentum must be numeric.")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int):
            raise TypeError("ELR state start_epoch must be an int.")
        if not isinstance(enabled, bool):
            raise TypeError("ELR state enabled must be bool.")
        if isinstance(epsilon, bool) or not isinstance(epsilon, float | int):
            raise TypeError("ELR state epsilon must be numeric.")
        parsed_momentum = float(target_momentum)
        parsed_epsilon = float(epsilon)
        if not 0.0 <= parsed_momentum < 1.0:
            raise ValueError("ELR state target_momentum must be in [0, 1).")
        if start_epoch < 0:
            raise ValueError("ELR state start_epoch must be non-negative.")
        if parsed_epsilon <= 0.0:
            raise ValueError("ELR state epsilon must be positive.")
        self.target_momentum = parsed_momentum
        self.start_epoch = start_epoch
        self.enabled = enabled
        self.epsilon = parsed_epsilon
