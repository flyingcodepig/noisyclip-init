"""Single-prototype builders for class-wise normalized image embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MeanPrototypeBuilder:
    """Build one L2-normalized arithmetic-mean prototype per class.

    The input `embeddings` must be a finite floating-point tensor shaped
    `[N, D]`; `targets` must be an int64 tensor shaped `[N]` with values in
    `[0, C)`. `sample_weights` is ignored by this builder and may be `None` or
    a finite `[N]` tensor. The output has shape `[C, D]` and unit L2 row norms.

    Raises:
        TypeError: If tensor dtypes are invalid.
        ValueError: If shapes, class coverage, target ranges, or values are
            invalid, or if any class mean has zero norm.
    """

    def fit(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """Return `[C, D]` L2-normalized arithmetic prototypes."""

        _validate_inputs(
            embeddings,
            targets,
            sample_weights,
            num_classes,
            require_weights=False,
        )
        prototypes = []
        for class_index in range(num_classes):
            class_embeddings = embeddings[targets == class_index]
            if class_embeddings.shape[0] == 0:
                raise ValueError(f"Missing samples for class target {class_index}.")
            prototypes.append(class_embeddings.mean(dim=0))
        return _normalize_prototypes(torch.stack(prototypes, dim=0))


@dataclass(frozen=True, slots=True)
class TrimmedMeanPrototypeBuilder:
    """Build one prototype per class after trimming farthest class samples.

    Args:
        keep_fraction: Fraction in `(0, 1]` of samples retained per class after
            ranking by distance to the preliminary class mean. At least one
            sample is kept in every class.

    Input shapes and output shapes match `MeanPrototypeBuilder`.

    Raises:
        ValueError: If `keep_fraction` is outside `(0, 1]`, a class is missing,
            or retained samples produce a zero-norm prototype.
        TypeError: If tensor dtypes are invalid.
    """

    keep_fraction: float = 0.8

    def __post_init__(self) -> None:
        """Validate the retained fraction range `(0, 1]`."""

        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {self.keep_fraction}.")

    def fit(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """Return `[C, D]` L2-normalized trimmed-mean prototypes."""

        _validate_inputs(
            embeddings,
            targets,
            sample_weights,
            num_classes,
            require_weights=False,
        )
        prototypes = []
        for class_index in range(num_classes):
            class_embeddings = embeddings[targets == class_index]
            if class_embeddings.shape[0] == 0:
                raise ValueError(f"Missing samples for class target {class_index}.")
            preliminary = class_embeddings.mean(dim=0, keepdim=True)
            distances = (class_embeddings - preliminary).norm(dim=1)
            keep_count = max(
                1,
                int(
                    torch.ceil(torch.tensor(class_embeddings.shape[0] * self.keep_fraction)).item()
                ),
            )
            retained_indices = torch.argsort(distances, stable=True)[:keep_count]
            prototypes.append(class_embeddings[retained_indices].mean(dim=0))
        return _normalize_prototypes(torch.stack(prototypes, dim=0))


@dataclass(frozen=True, slots=True)
class WeightedMeanPrototypeBuilder:
    """Build one sample-weighted mean prototype per class.

    The input `sample_weights` must be a finite floating-point tensor shaped
    `[N]` with values in `[0, +inf)`. For every class, the sum of in-class
    weights must be strictly positive. Output prototypes have shape `[C, D]`
    and L2-normalized rows.

    Raises:
        TypeError: If embeddings or weights are not floating-point tensors, or
            targets is not int64.
        ValueError: If shapes, class coverage, target ranges, finite values, or
            per-class effective weights are invalid.
    """

    def fit(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """Return `[C, D]` L2-normalized weighted-mean prototypes."""

        _validate_inputs(
            embeddings,
            targets,
            sample_weights,
            num_classes,
            require_weights=True,
        )
        if sample_weights is None:
            raise ValueError("sample_weights is required for weighted mean prototypes.")
        prototypes = []
        for class_index in range(num_classes):
            mask = targets == class_index
            if not bool(mask.any()):
                raise ValueError(f"Missing samples for class target {class_index}.")
            class_weights = sample_weights[mask]
            weight_sum = class_weights.sum()
            if not bool(weight_sum > 0):
                raise ValueError(f"Effective sample weight is zero for class target {class_index}.")
            class_embeddings = embeddings[mask]
            prototypes.append((class_embeddings * class_weights[:, None]).sum(dim=0) / weight_sum)
        return _normalize_prototypes(torch.stack(prototypes, dim=0))


def build_prototype_builder(
    method: Literal["mean", "trimmed_mean", "weighted_mean"],
    *,
    keep_fraction: float = 0.8,
) -> MeanPrototypeBuilder | TrimmedMeanPrototypeBuilder | WeightedMeanPrototypeBuilder:
    """Construct a prototype builder by method name.

    Args:
        method: One of `mean`, `trimmed_mean`, or `weighted_mean`.
        keep_fraction: Retained fraction used only by `trimmed_mean`.

    Returns:
        A concrete builder whose `fit` method maps `[N, D]` embeddings and
        `[N]` targets to `[C, D]` L2-normalized prototypes.

    Raises:
        ValueError: If `method` is unsupported or `keep_fraction` is invalid.
    """

    if method == "mean":
        return MeanPrototypeBuilder()
    if method == "trimmed_mean":
        return TrimmedMeanPrototypeBuilder(keep_fraction=keep_fraction)
    if method == "weighted_mean":
        return WeightedMeanPrototypeBuilder()
    raise ValueError(f"Unsupported prototype builder method: {method!r}.")


def _validate_inputs(
    embeddings: Tensor,
    targets: Tensor,
    sample_weights: Tensor | None,
    num_classes: int,
    *,
    require_weights: bool,
) -> None:
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}.")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}.")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have positive N and D dimensions.")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must be a floating-point tensor.")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contains NaN or Inf values.")
    if targets.ndim != 1 or targets.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"targets must have shape [N] matching embeddings, got {tuple(targets.shape)}."
        )
    if targets.dtype != torch.int64:
        raise TypeError("targets must be an int64 tensor.")
    if not torch.isfinite(targets.to(torch.float32)).all():
        raise ValueError("targets contains invalid numeric values.")
    if bool((targets < 0).any()) or bool((targets >= num_classes).any()):
        raise ValueError(f"targets must be in [0, {num_classes}).")
    if sample_weights is None:
        if require_weights:
            raise ValueError("sample_weights is required.")
        return
    if sample_weights.ndim != 1 or sample_weights.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "sample_weights must have shape [N] matching embeddings, "
            f"got {tuple(sample_weights.shape)}."
        )
    if not sample_weights.is_floating_point():
        raise TypeError("sample_weights must be a floating-point tensor.")
    if not torch.isfinite(sample_weights).all():
        raise ValueError("sample_weights contains NaN or Inf values.")
    if bool((sample_weights < 0).any()):
        raise ValueError("sample_weights must be non-negative.")


def _normalize_prototypes(prototypes: Tensor) -> Tensor:
    norms = prototypes.norm(dim=1)
    if not torch.isfinite(norms).all():
        raise ValueError("prototype norms contain NaN or Inf values.")
    if bool((norms <= 0).any()):
        raise ValueError("class prototype has zero norm.")
    return F.normalize(prototypes, dim=1)
