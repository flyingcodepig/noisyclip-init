"""Weak-to-strong prediction consistency loss."""

from __future__ import annotations

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


class ConsistencyLoss:
    """KL consistency from detached weak predictions to strong predictions.

    Args:
        temperature: Positive softmax temperature.
        start_epoch: First epoch that contributes a non-zero scalar.
        enabled: When false, the term returns zero and does not require a strong
            view.

    Inputs:
        Weak and strong logits must both be finite `[B, C]` with identical
        shape. When enabled, `student_strong` must be present even during warmup
        so data-pipeline mistakes fail early. The weak target distribution is
        detached before the KL computation.

    Outputs:
        A finite scalar KL divergence. Zero is returned when disabled or before
        `start_epoch`.

    Raises:
        ValueError: If strong output is missing while enabled, shapes differ,
            epoch is negative, or tensors are non-finite.
    """

    name = "loss/consistency"

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        start_epoch: int = 0,
        enabled: bool = True,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative.")
        self.temperature = temperature
        self.start_epoch = start_epoch
        self.enabled = enabled

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> Tensor:
        """Compute weak-target to strong-prediction KL consistency.

        Args:
            batch: Batch with unique sample IDs.
            student_weak: Weak-view output with logits `[B, C]`.
            student_strong: Strong-view output with logits `[B, C]` when enabled.
            sample_states: Batch-aligned states providing trust/curriculum weights.
            epoch: Non-negative epoch index controlling warmup.

        Returns:
            A finite scalar tensor.

        Raises:
            ValueError: If the strong output is missing while enabled, epoch is
                negative, shapes mismatch, or values are non-finite.
        """

        batch_size, num_classes = require_model_output("student_weak", student_weak)
        require_batch_alignment(batch, batch_size, sample_states)
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        if not self.enabled:
            zero = student_weak.logits.new_zeros(())
            require_scalar(self.name, zero)
            return zero
        if student_strong is None:
            raise ValueError("student_strong is required when consistency loss is enabled.")

        strong_batch_size, strong_num_classes = require_model_output(
            "student_strong", student_strong
        )
        if (strong_batch_size, strong_num_classes) != (batch_size, num_classes):
            raise ValueError(
                "student_strong.logits must match student_weak.logits shape "
                f"[{batch_size}, {num_classes}], got {tuple(student_strong.logits.shape)}."
            )
        if epoch < self.start_epoch:
            zero = student_weak.logits.new_zeros(())
            require_scalar(self.name, zero)
            return zero

        target = F.softmax(student_weak.logits / self.temperature, dim=1).detach()
        log_prediction = F.log_softmax(student_strong.logits / self.temperature, dim=1)
        per_sample = F.kl_div(log_prediction, target, reduction="none").sum(dim=1)
        weights = supervised_weights(
            sample_states,
            device=per_sample.device,
            dtype=per_sample.dtype,
            require_positive=False,
        )
        denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        loss = (per_sample * weights).sum() / denominator * (self.temperature**2)
        require_scalar(self.name, loss)
        return loss
