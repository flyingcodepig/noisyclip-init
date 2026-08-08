"""Linear and cosine classifier heads for normalized CLIP embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from noisyclip.models.validation import require_embedding_batch
from noisyclip.utils.runtime_checks import value_checks_enabled


class LinearClassifierHead(nn.Module):
    """Linear classifier returning unnormalized `[B, C]` logits.

    Args:
        embedding_dim: Input embedding dimension `D`; must be positive.
        num_classes: Number of classes `C`; must be positive.

    Raises:
        ValueError: If dimensions are not positive or forward input is invalid.
        TypeError: If forward input is not floating point.
    """

    embedding_dim: int
    num_classes: int

    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        _validate_positive_dimensions(embedding_dim=embedding_dim, num_classes=num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Return `[B, C]` logits from finite `[B, D]` embeddings."""

        require_embedding_batch(embeddings, embedding_dim=self.embedding_dim)
        logits = self.linear(embeddings)
        _validate_logits(logits, num_classes=self.num_classes)
        return logits


class CosineClassifierHead(nn.Module):
    """Cosine classifier with learnable bounded logit temperature.

    Args:
        embedding_dim: Input embedding dimension `D`; must be positive.
        num_classes: Number of classes `C`; must be positive.
        temperature_init: Initial multiplicative logit scale; finite and within
            `[temperature_min, temperature_max]`.
        temperature_min: Lower inclusive bound for temperature.
        temperature_max: Upper inclusive bound for temperature.

    Raises:
        ValueError: If dimensions, temperature bounds, or forward inputs are
            invalid.
        TypeError: If forward input is not floating point.
    """

    embedding_dim: int
    num_classes: int

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_classes: int,
        temperature_init: float,
        temperature_min: float,
        temperature_max: float,
    ) -> None:
        super().__init__()
        _validate_positive_dimensions(embedding_dim=embedding_dim, num_classes=num_classes)
        _validate_temperature_bounds(
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
        )
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def current_temperature(self) -> Tensor:
        """Return the scalar temperature clamped to configured bounds.

        Returns:
            Scalar tensor in `[temperature_min, temperature_max]`.
        """

        return self.temperature.clamp(self.temperature_min, self.temperature_max)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Return unsoftmaxed cosine logits shaped `[B, C]`.

        Both input embeddings and class weights are L2-normalized inside the
        method; input must be finite `[B, D]`, output is finite `[B, C]`.
        """

        require_embedding_batch(embeddings, embedding_dim=self.embedding_dim)
        normalized_embeddings = F.normalize(embeddings.float(), dim=1)
        normalized_weight = F.normalize(self.weight.float(), dim=1)
        logits = self.current_temperature() * normalized_embeddings @ normalized_weight.t()
        _validate_logits(logits, num_classes=self.num_classes)
        return logits


def build_classifier_head(
    *,
    head_type: str,
    embedding_dim: int,
    num_classes: int,
    temperature_init: float | None = None,
    temperature_min: float | None = None,
    temperature_max: float | None = None,
) -> nn.Module:
    """Build a linear or cosine classifier head from primitive config fields.

    Args:
        head_type: Either `linear` or `cosine`.
        embedding_dim: Input dimension `D`.
        num_classes: Class count `C`.
        temperature_init: Required for cosine heads.
        temperature_min: Required for cosine heads.
        temperature_max: Required for cosine heads.

    Returns:
        A classifier head whose forward maps `[B, D]` to `[B, C]` logits.

    Raises:
        ValueError: If the head type or cosine temperature fields are invalid.
    """

    if head_type == "linear":
        return LinearClassifierHead(embedding_dim=embedding_dim, num_classes=num_classes)
    if head_type == "cosine":
        if temperature_init is None or temperature_min is None or temperature_max is None:
            raise ValueError(
                "Cosine head requires temperature_init, temperature_min, and temperature_max."
            )
        return CosineClassifierHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
        )
    raise ValueError(f"Unsupported classifier head type: {head_type!r}.")


def _validate_positive_dimensions(*, embedding_dim: int, num_classes: int) -> None:
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")


def _validate_temperature_bounds(
    *,
    temperature_init: float,
    temperature_min: float,
    temperature_max: float,
) -> None:
    values = {
        "temperature_init": temperature_init,
        "temperature_min": temperature_min,
        "temperature_max": temperature_max,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if temperature_min > temperature_max:
        raise ValueError("temperature_min cannot exceed temperature_max.")
    if not temperature_min <= temperature_init <= temperature_max:
        raise ValueError("temperature_init must be inside [temperature_min, temperature_max].")


def _validate_logits(logits: Tensor, *, num_classes: int) -> None:
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [B, {num_classes}], got {tuple(logits.shape)}.")
    if value_checks_enabled() and not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf values.")
