"""Tensor and model validation helpers for NoisyCLIP model components."""

from __future__ import annotations

import torch
from torch import Tensor


def require_image_batch(images: Tensor, *, field_name: str = "images") -> None:
    """Validate an image batch shaped `[B, 3, 224, 224]`.

    Args:
        images: Floating-point tensor with shape `[B, 3, 224, 224]`; `B` must
            be positive and all values must be finite.
        field_name: Name included in validation errors.

    Raises:
        TypeError: If `images` is not floating point.
        ValueError: If shape, batch size, or values are invalid.
    """

    if images.ndim != 4:
        raise ValueError(
            f"{field_name} must have shape [B, 3, 224, 224], got {tuple(images.shape)}."
        )
    if images.shape[0] <= 0:
        raise ValueError(f"{field_name} must contain a non-empty batch.")
    if images.shape[1:] != (3, 224, 224):
        raise ValueError(
            f"{field_name} must have shape [B, 3, 224, 224], got {tuple(images.shape)}."
        )
    if not images.is_floating_point():
        raise TypeError(f"{field_name} must be a floating-point tensor.")
    if not torch.isfinite(images).all():
        raise ValueError(f"{field_name} contains NaN or Inf values.")


def require_embedding_batch(
    embeddings: Tensor,
    *,
    embedding_dim: int,
    field_name: str = "embeddings",
) -> None:
    """Validate an embedding batch shaped `[B, D]`.

    Args:
        embeddings: Floating-point tensor with shape `[B, D]`, positive `B`,
            expected `D`, and finite values.
        embedding_dim: Required embedding dimension `D`.
        field_name: Name included in validation errors.

    Raises:
        TypeError: If `embeddings` is not floating point.
        ValueError: If shape, batch size, dimension, or values are invalid.
    """

    if embeddings.ndim != 2:
        raise ValueError(f"{field_name} must have shape [B, D], got {tuple(embeddings.shape)}.")
    if embeddings.shape[0] <= 0:
        raise ValueError(f"{field_name} must contain a non-empty batch.")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"{field_name} dimension mismatch: expected D={embedding_dim}, "
            f"got {embeddings.shape[1]}."
        )
    if not embeddings.is_floating_point():
        raise TypeError(f"{field_name} must be a floating-point tensor.")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{field_name} contains NaN or Inf values.")


def require_l2_normalized(
    embeddings: Tensor,
    *,
    atol: float = 1e-4,
    field_name: str = "embeddings",
) -> None:
    """Validate that `[B, D]` embeddings have unit L2 norm within tolerance.

    Args:
        embeddings: Floating-point tensor with shape `[B, D]`; rows must have
            finite norms close to `1`.
        atol: Absolute tolerance for each row norm.
        field_name: Name included in validation errors.

    Raises:
        ValueError: If any row is non-finite or not L2-normalized.
    """

    norms = embeddings.norm(dim=1)
    if not torch.isfinite(norms).all():
        raise ValueError(f"{field_name} norm contains NaN or Inf values.")
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=atol):
        raise ValueError(f"{field_name} must be L2-normalized row-wise.")


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module to inspect.
        trainable_only: When true, only parameters with `requires_grad=True`
            are counted.

    Returns:
        Number of scalar parameters as an integer.
    """

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )
