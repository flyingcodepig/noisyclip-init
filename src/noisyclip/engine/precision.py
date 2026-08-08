"""AMP, gradient accumulation, clipping, and finite-loss/gradient guards."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


class NonFiniteTrainingError(RuntimeError):
    """Raised when loss or gradients contain NaN/Inf values."""


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Runtime precision and optimizer-step settings.

    Attributes:
        precision: `fp32` or `amp_fp16`.
        gradient_accumulation_steps: Positive number of microbatches per step.
        gradient_clip_norm: Positive max norm used before optimizer stepping.
    """

    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate precision, accumulation, and clip ranges."""

        if self.precision not in {"fp32", "amp_fp16"}:
            raise ValueError(f"Unsupported precision: {self.precision!r}.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")


class PrecisionManager:
    """Manage autocast, GradScaler, accumulation, and gradient safety checks.

    Args:
        config: Precision/accumulation/clipping settings.
        device: Torch device used for training. CPU always runs an fp32-safe
            path even when `amp_fp16` is requested.

    Raises:
        ValueError: If config fields are invalid.
    """

    def __init__(self, config: PrecisionConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.use_amp = config.precision == "amp_fp16" and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._pending_microbatches = 0

    def autocast(self) -> Any:
        """Return an autocast context manager for forward/loss computation."""

        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward(self, loss: Tensor) -> None:
        """Scale accumulation loss and run backward after finite validation.

        Args:
            loss: Scalar tensor before accumulation scaling.

        Raises:
            NonFiniteTrainingError: If `loss` is not finite.
            ValueError: If `loss` is not scalar.
        """

        if loss.ndim != 0:
            raise ValueError(f"loss must be scalar, got shape {tuple(loss.shape)}.")
        if not torch.isfinite(loss).item():
            raise NonFiniteTrainingError("Training loss contains NaN or Inf.")
        self.scaler.scale(loss).backward()
        self._pending_microbatches += 1

    def step_if_needed(
        self,
        *,
        microbatch_index: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        force: bool = False,
        gradient_validator: Callable[[], None] | None = None,
    ) -> bool:
        """Run an optimizer step at accumulation boundaries.

        Args:
            microbatch_index: Zero-based microbatch index within the epoch.
            model: Model whose gradients are checked/clipped.
            optimizer: Optimizer stepped at accumulation boundaries.

        Returns:
            `True` when an optimizer step occurred, otherwise `False`.

        Raises:
            NonFiniteTrainingError: If any trainable gradient is NaN or Inf.
        """

        del microbatch_index
        if self._pending_microbatches == 0:
            return False
        if not force and self._pending_microbatches < self.config.gradient_accumulation_steps:
            return False
        self.scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self._pending_microbatches)
        if gradient_validator is not None:
            gradient_validator()
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.gradient_clip_norm,
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as exc:
            raise NonFiniteTrainingError("Gradient contains NaN or Inf values.") from exc
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        self._pending_microbatches = 0
        return True

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable GradScaler state mapping."""

        if self._pending_microbatches:
            raise RuntimeError("Cannot checkpoint with unstepped accumulated gradients.")
        return {"use_amp": self.use_amp, "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore GradScaler state produced by `state_dict`.

        Args:
            state_dict: Mapping containing a `scaler` entry.

        Raises:
            TypeError: If `scaler` is not a mapping.
        """

        scaler_state = state_dict.get("scaler", {})
        if not isinstance(scaler_state, dict):
            raise TypeError("precision scaler state must be a dictionary.")
        self.scaler.load_state_dict(scaler_state)
        self._pending_microbatches = 0


def check_tensor_finite(name: str, tensor: Tensor) -> None:
    """Require all values in `tensor` to be finite.

    Args:
        name: Diagnostic tensor name.
        tensor: Tensor of any shape or dtype.

    Raises:
        NonFiniteTrainingError: If any element is NaN or Inf.
    """

    if not torch.isfinite(tensor.detach()).all().item():
        raise NonFiniteTrainingError(f"{name} contains NaN or Inf values.")


def check_gradients_finite(parameters: Iterable[nn.Parameter]) -> None:
    """Require every existing gradient to be finite.

    Args:
        parameters: Iterable of model parameters.

    Raises:
        NonFiniteTrainingError: If any gradient contains NaN or Inf.
    """

    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise NonFiniteTrainingError("Gradient contains NaN or Inf values.")
