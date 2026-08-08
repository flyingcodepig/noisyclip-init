"""Shared validation helpers for loss implementations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState
from noisyclip.utils.runtime_checks import value_checks_enabled


def require_floating_tensor(name: str, value: Tensor, shape: tuple[int | None, ...]) -> None:
    """Validate a finite floating tensor against a rank and optional dimensions.

    Args:
        name: Human-readable field name used in error messages.
        value: Tensor to validate; it must be floating, finite, and match `shape`.
        shape: Expected shape where `None` accepts any size at that dimension.

    Raises:
        TypeError: If `value` is not floating point.
        ValueError: If rank, dimensions, or finite-value checks fail.
    """

    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating tensor.")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got shape {tuple(value.shape)}.")
    for dim, expected in enumerate(shape):
        if expected is not None and value.shape[dim] != expected:
            raise ValueError(f"{name} dimension {dim} must be {expected}, got {value.shape[dim]}.")
    if value_checks_enabled() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


def require_scalar(name: str, value: Tensor) -> None:
    """Validate that `value` is a finite scalar tensor.

    Args:
        name: Human-readable field name used in error messages.
        value: Tensor expected to have shape `[]` and finite value.

    Raises:
        ValueError: If the tensor is not scalar or contains NaN/Inf.
    """

    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
    if value_checks_enabled() and not torch.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def require_model_output(name: str, output: ModelOutput) -> tuple[int, int]:
    """Validate model logits and embeddings and return `(batch_size, num_classes)`.

    Args:
        name: Prefix used in error messages.
        output: Model output with logits `[B, C]` and embedding `[B, D]`.

    Returns:
        A tuple containing batch size and number of classes.

    Raises:
        TypeError: If logits or embeddings are not floating point.
        ValueError: If shapes are invalid or tensors contain NaN/Inf.
    """

    require_floating_tensor(f"{name}.logits", output.logits, (None, None))
    require_floating_tensor(f"{name}.embedding", output.embedding, (output.logits.shape[0], None))
    if output.logits.shape[1] <= 0:
        raise ValueError(f"{name}.logits must contain at least one class.")
    return int(output.logits.shape[0]), int(output.logits.shape[1])


def require_batch_alignment(
    batch: Batch,
    batch_size: int,
    sample_states: list[SampleState] | None,
) -> None:
    """Validate batch IDs, target length, and optional state alignment.

    Args:
        batch: Batch whose `sample_ids` length must equal `batch_size`.
        batch_size: Expected number of samples from model logits.
        sample_states: Optional per-sample states expected in batch order.

    Raises:
        ValueError: If lengths differ, IDs are duplicated, or states are misaligned.
    """

    if len(batch.sample_ids) != batch_size:
        raise ValueError(
            f"batch.sample_ids length must be {batch_size}, got {len(batch.sample_ids)}."
        )
    if len(set(batch.sample_ids)) != len(batch.sample_ids):
        raise ValueError("batch.sample_ids must be unique within a batch.")
    if batch.targets is not None and batch.targets.shape != (batch_size,):
        raise ValueError(
            f"batch.targets must have shape [{batch_size}], got {tuple(batch.targets.shape)}."
        )
    if sample_states is None:
        return
    if len(sample_states) != batch_size:
        raise ValueError(f"sample_states length must be {batch_size}, got {len(sample_states)}.")
    state_ids = [state.sample_id for state in sample_states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("sample_states sample_id values must be unique within a batch.")
    if state_ids != batch.sample_ids:
        raise ValueError("sample_states must be ordered to match batch.sample_ids.")


def require_targets(targets: Tensor | None, batch_size: int, num_classes: int) -> Tensor:
    """Validate supervised targets and return the non-optional tensor.

    Args:
        targets: Tensor with shape `[B]`, dtype int64, and values in `[0, C)`.
        batch_size: Expected `B`.
        num_classes: Number of classes `C`.

    Returns:
        The validated target tensor.

    Raises:
        ValueError: If targets are missing, wrong-shaped, or out of range.
        TypeError: If dtype is not `torch.int64`.
    """

    if targets is None:
        raise ValueError("batch.targets is required for supervised cross entropy.")
    if targets.dtype != torch.int64:
        raise TypeError(f"batch.targets must have dtype torch.int64, got {targets.dtype}.")
    if targets.shape != (batch_size,):
        raise ValueError(
            f"batch.targets must have shape [{batch_size}], got {tuple(targets.shape)}."
        )
    if batch_size > 0 and value_checks_enabled():
        min_target = int(targets.min().item())
        max_target = int(targets.max().item())
        if min_target < 0 or max_target >= num_classes:
            raise ValueError(
                "batch.targets must be in range [0, "
                f"{num_classes}), got min={min_target}, max={max_target}."
            )
    return targets


def supervised_weights(
    sample_states: list[SampleState],
    *,
    device: torch.device,
    dtype: torch.dtype,
    require_positive: bool,
) -> Tensor:
    """Return validated `[B]` supervised weights from sample state.

    Args:
        sample_states: Batch-aligned states with `supervised_weight` in `[0, 1]`.
        device: Device for the returned tensor.
        dtype: Floating dtype for the returned tensor.
        require_positive: Whether a zero weight sum should raise.

    Returns:
        A finite `[B]` floating tensor on `device`.

    Raises:
        ValueError: If any weight is non-finite, outside `[0, 1]`, or all-zero when
            `require_positive` is true.
    """

    values = [state.supervised_weight for state in sample_states]
    weights = torch.tensor(values, device=device, dtype=dtype)
    if weights.ndim != 1:
        raise ValueError("supervised weights must form a rank-1 tensor.")
    if value_checks_enabled() and not torch.isfinite(weights).all():
        raise ValueError("SampleState.supervised_weight must be finite.")
    if value_checks_enabled() and bool((weights < 0).any() or (weights > 1).any()):
        raise ValueError("SampleState.supervised_weight must be in the range [0, 1].")
    if require_positive and value_checks_enabled() and float(weights.sum().item()) <= 0.0:
        raise ValueError("At least one SampleState.supervised_weight must be positive.")
    return weights


def require_normalized_embeddings(name: str, embeddings: Tensor, tolerance: float) -> None:
    """Validate that embeddings are finite and L2-normalized along dimension 1.

    Args:
        name: Human-readable field name used in error messages.
        embeddings: Floating tensor shaped `[B, D]`.
        tolerance: Absolute tolerance for the unit-norm check.

    Raises:
        ValueError: If shape, finite values, or L2 norms are invalid.
    """

    require_floating_tensor(name, embeddings, (None, None))
    if not value_checks_enabled():
        return
    norms = embeddings.norm(dim=1)
    if not torch.isfinite(norms).all():
        raise ValueError(f"{name} norms must be finite.")
    if not torch.allclose(norms, torch.ones_like(norms), atol=tolerance, rtol=0.0):
        raise ValueError(f"{name} must be L2-normalized within atol={tolerance}.")


def clone_component_mapping(components: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Validate and copy scalar finite component values by stable name.

    Args:
        components: Mapping from component name to scalar tensor.

    Returns:
        A regular dictionary with the same scalar tensors.

    Raises:
        ValueError: If a name is empty, a value is not scalar, or any value is non-finite.
    """

    copied: dict[str, Tensor] = {}
    for name, value in components.items():
        if not name:
            raise ValueError("Loss component names must be non-empty.")
        require_scalar(name, value)
        copied[name] = value
    return copied
