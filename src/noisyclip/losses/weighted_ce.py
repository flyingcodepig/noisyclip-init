"""Weighted supervised cross-entropy loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.losses._validation import (
    require_batch_alignment,
    require_model_output,
    require_scalar,
    require_targets,
    supervised_weights,
)
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState
from noisyclip.utils.runtime_checks import value_checks_enabled


class WeightedCrossEntropyLoss:
    """Per-sample cross entropy normalized by effective supervised weight.

    Args:
        label_smoothing: Smoothing value in `[0, 1)`, forwarded to PyTorch cross
            entropy. `0` keeps one-hot targets.

    Inputs:
        `student_weak.logits` must be finite `[B, C]`; `batch.targets` must be
        int64 `[B]` with values in `[0, C)`; `sample_states` must be ordered to
        match `batch.sample_ids` and contain finite weights in `[0, 1]`.

    Outputs:
        Returns `(loss, per_sample_loss)`, where `loss` is scalar and
        `per_sample_loss` is finite `[B]` before detach so the composite loss can
        safely detach it for state updates.

    Raises:
        ValueError: On shape mismatch, duplicate IDs, target range errors,
            non-finite logits/losses, or all-zero supervised weights.
        TypeError: On invalid target or logit dtype.
    """

    name = "loss/ce"

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1).")
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        sample_states: list[SampleState],
    ) -> tuple[Tensor, Tensor]:
        """Compute scalar weighted CE and finite `[B]` per-sample CE.

        Args:
            batch: Batch with unique `sample_ids` and int64 targets `[B]`.
            student_weak: Student output with finite logits `[B, C]`.
            sample_states: Batch-aligned state list carrying supervised weights.

        Returns:
            A scalar tensor normalized by `sum(supervised_weight)` and an
            unweighted per-sample CE tensor shaped `[B]`.

        Raises:
            ValueError: If inputs are misaligned, out of range, all-zero, or
                non-finite.
            TypeError: If supervised targets are not int64.
        """

        batch_size, num_classes = require_model_output("student_weak", student_weak)
        require_batch_alignment(batch, batch_size, sample_states)
        targets = require_targets(batch.targets, batch_size, num_classes)
        weights = supervised_weights(
            sample_states,
            device=student_weak.logits.device,
            dtype=student_weak.logits.dtype,
            require_positive=True,
        )

        per_sample = F.cross_entropy(
            student_weak.logits,
            targets.to(device=student_weak.logits.device),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if per_sample.shape != (batch_size,):
            raise ValueError(
                f"per-sample cross entropy must have shape [{batch_size}], "
                f"got {tuple(per_sample.shape)}."
            )
        if value_checks_enabled() and not torch.isfinite(per_sample).all():
            raise ValueError("per-sample cross entropy must be finite.")

        loss = (per_sample * weights).sum() / weights.sum()
        require_scalar(self.name, loss)
        return loss, per_sample
