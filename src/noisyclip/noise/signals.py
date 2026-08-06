"""Raw per-sample trust signals for noisy-label training."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


@dataclass(frozen=True, slots=True)
class EmaLossSignal:
    """Compute per-sample exponential moving average cross-entropy loss.

    Args:
        momentum: EMA coefficient in `[0, 1)`. The previous `SampleState`
            `ema_loss` is blended with the current per-sample CE.

    Inputs use `Batch.targets` shaped `[B]` and `output_weak.logits` shaped
    `[B, C]`. The returned tensor has shape `[B]`, is finite, and is not
    normalized. Lower values indicate more trustworthy samples.

    Raises:
        ValueError: If targets are missing, shapes mismatch, values are
            non-finite, or `state` length differs from batch size.
    """

    momentum: float = 0.9
    name: str = "ema_loss"

    def __post_init__(self) -> None:
        """Validate the EMA momentum range `[0, 1)`."""

        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}.")

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` EMA loss values without class normalization."""

        del output_strong, prototypes
        _validate_batch_size(batch, output_weak, state)
        if batch.targets is None:
            raise ValueError("ema_loss requires batch.targets shaped [B].")
        _validate_targets(batch.targets, output_weak.logits.shape[0], output_weak.logits.shape[1])
        losses = F.cross_entropy(output_weak.logits, batch.targets, reduction="none")
        previous = torch.tensor(
            [item.ema_loss for item in state],
            dtype=losses.dtype,
            device=losses.device,
        )
        values = self.momentum * previous + (1.0 - self.momentum) * losses
        _require_finite_vector(values, field_name=self.name)
        return values


@dataclass(frozen=True, slots=True)
class AugmentationAgreementSignal:
    """Compute weak/strong probability agreement for each sample.

    Inputs require `output_weak.logits` and `output_strong.logits` shaped
    `[B, C]`. The returned `[B]` tensor is the dot product between weak and
    strong softmax probabilities, so values lie in `[0, 1]`. Higher values are
    more trustworthy.

    Raises:
        ValueError: If `output_strong` is missing, shapes mismatch, or values
            are non-finite.
    """

    name: str = "augmentation_agreement"

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` weak/strong agreement values."""

        del prototypes
        _validate_batch_size(batch, output_weak, state)
        if output_strong is None:
            raise ValueError("augmentation_agreement requires output_strong.")
        if output_strong.logits.shape != output_weak.logits.shape:
            raise ValueError(
                "output_strong.logits must match output_weak.logits shape "
                f"{tuple(output_weak.logits.shape)}, got {tuple(output_strong.logits.shape)}."
            )
        weak_probs = output_weak.logits.softmax(dim=1)
        strong_probs = output_strong.logits.softmax(dim=1)
        values = (weak_probs * strong_probs).sum(dim=1)
        _require_finite_vector(values, field_name=self.name)
        return values


@dataclass(frozen=True, slots=True)
class PrototypeSimilaritySignal:
    """Compute cosine similarity to the labeled class prototype.

    Inputs require `output_weak.embedding` shaped `[B, D]`, `prototypes` shaped
    `[C, D]`, and `Batch.targets` shaped `[B]`. The returned raw `[B]` values
    are cosine similarities in `[-1, 1]`. Higher values are more trustworthy.

    Raises:
        ValueError: If prototypes or targets are missing, shapes/ranges are
            invalid, or values are non-finite.
    """

    name: str = "prototype_similarity"

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` target-prototype cosine similarities."""

        del output_strong
        _validate_batch_size(batch, output_weak, state)
        prototypes = _validate_prototype_inputs(batch, output_weak, prototypes)
        targets = batch.targets
        if targets is None:
            raise ValueError("prototype_similarity requires batch.targets shaped [B].")
        similarities = output_weak.embedding @ prototypes.T
        values = similarities.gather(1, targets[:, None]).squeeze(1)
        _require_finite_vector(values, field_name=self.name)
        return values


@dataclass(frozen=True, slots=True)
class PrototypeMarginSignal:
    """Compute target prototype similarity minus strongest non-target match.

    Inputs match `PrototypeSimilaritySignal`, with at least two classes. The
    returned raw `[B]` values are margins in `[-2, 2]`. Higher values are more
    trustworthy.

    Raises:
        ValueError: If prototypes/targets are missing, fewer than two classes
            exist, shapes/ranges are invalid, or values are non-finite.
    """

    name: str = "prototype_margin"

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` target-vs-nearest-other prototype margins."""

        del output_strong
        _validate_batch_size(batch, output_weak, state)
        prototypes = _validate_prototype_inputs(batch, output_weak, prototypes)
        if prototypes.shape[0] < 2:
            raise ValueError("prototype_margin requires at least two classes.")
        targets = batch.targets
        if targets is None:
            raise ValueError("prototype_margin requires batch.targets shaped [B].")
        similarities = output_weak.embedding @ prototypes.T
        target_scores = similarities.gather(1, targets[:, None]).squeeze(1)
        masked = similarities.clone()
        masked.scatter_(1, targets[:, None], float("-inf"))
        nearest_other = masked.max(dim=1).values
        values = target_scores - nearest_other
        _require_finite_vector(values, field_name=self.name)
        return values


@dataclass(frozen=True, slots=True)
class PredictionStabilitySignal:
    """Compare current probabilities with previous EMA probabilities.

    Inputs use `output_weak.logits` shaped `[B, C]` and the per-sample state
    list. When a sample has no previous `ema_probs`, the raw stability is `0`.
    Otherwise the returned `[B]` value is the probability dot product in
    `[0, 1]`. Higher values are more trustworthy.

    Raises:
        ValueError: If previous probability lengths mismatch `C`, shapes are
            invalid, or values are non-finite.
    """

    name: str = "prediction_stability"

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` prediction-stability values."""

        del output_strong, prototypes
        _validate_batch_size(batch, output_weak, state)
        probs = output_weak.logits.softmax(dim=1)
        previous_rows: list[Tensor] = []
        for index, item in enumerate(state):
            if item.ema_probs is None:
                previous_rows.append(torch.zeros_like(probs[index]))
                continue
            if len(item.ema_probs) != probs.shape[1]:
                raise ValueError(
                    "ema_probs length must match logits class dimension for "
                    f"sample_id={item.sample_id}."
                )
            previous = torch.tensor(item.ema_probs, dtype=probs.dtype, device=probs.device)
            previous_rows.append(previous)
        previous_probs = torch.stack(previous_rows, dim=0)
        values = (probs * previous_probs).sum(dim=1)
        _require_finite_vector(values, field_name=self.name)
        return values


def update_prediction_history(
    states: list[SampleState],
    logits: Tensor,
    *,
    epoch: int,
    momentum: float = 0.9,
) -> list[SampleState]:
    """Update per-sample probability EMA and observation count once per epoch.

    Args:
        states: Unique sample states aligned with logits rows.
        logits: Finite model logits shaped `[N, C]`.
        epoch: Non-negative epoch recorded in returned states.
        momentum: Probability EMA coefficient in `[0, 1)`.

    Returns:
        New states preserving input order, with normalized `ema_probs`,
        `seen_count + 1`, and `updated_epoch=epoch`.

    Raises:
        ValueError: If IDs, shapes, epoch, momentum, previous probability
            vectors, or logits are invalid.
    """

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    if logits.ndim != 2 or logits.shape[0] != len(states) or logits.shape[1] <= 0:
        raise ValueError(
            f"logits must have shape [{len(states)}, C] with C > 0, got {tuple(logits.shape)}."
        )
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite floating-point tensor.")
    sample_ids = [state.sample_id for state in states]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("states must contain unique, non-empty sample_id values.")

    probabilities = logits.detach().softmax(dim=1).to(dtype=torch.float32, device="cpu")
    updated: list[SampleState] = []
    for index, state in enumerate(states):
        current = probabilities[index]
        if state.ema_probs is None:
            history = current
        else:
            if len(state.ema_probs) != logits.shape[1]:
                raise ValueError(
                    "ema_probs length must match logits class dimension for "
                    f"sample_id={state.sample_id}."
                )
            previous = torch.tensor(state.ema_probs, dtype=torch.float32)
            if not torch.isfinite(previous).all() or bool(
                (previous < 0).any() or (previous > 1).any()
            ):
                raise ValueError(f"ema_probs is invalid for sample_id={state.sample_id}.")
            probability_sum = float(previous.sum().item())
            if abs(probability_sum - 1.0) > 1e-4:
                raise ValueError(f"ema_probs must sum to 1 for sample_id={state.sample_id}.")
            history = momentum * previous + (1.0 - momentum) * current
        history = history / history.sum()
        updated.append(
            replace(
                state,
                seen_count=state.seen_count + 1,
                ema_probs=history.tolist(),
                updated_epoch=epoch,
            )
        )
    return updated


def _validate_batch_size(batch: Batch, output_weak: ModelOutput, state: list[SampleState]) -> None:
    if output_weak.logits.ndim != 2:
        raise ValueError(
            f"output_weak.logits must have shape [B, C], got {tuple(output_weak.logits.shape)}."
        )
    batch_size = output_weak.logits.shape[0]
    if len(batch.sample_ids) != batch_size:
        raise ValueError("batch.sample_ids length must match logits batch dimension.")
    if len(set(batch.sample_ids)) != len(batch.sample_ids):
        raise ValueError("batch.sample_ids contains duplicate sample_id values.")
    if len(state) != batch_size:
        raise ValueError("state length must match logits batch dimension.")
    if [item.sample_id for item in state] != batch.sample_ids:
        raise ValueError("state order must match batch.sample_ids.")
    if not torch.isfinite(output_weak.logits).all():
        raise ValueError("output_weak.logits contains NaN or Inf values.")


def _validate_targets(targets: Tensor, batch_size: int, num_classes: int) -> None:
    if targets.ndim != 1 or targets.shape[0] != batch_size:
        raise ValueError(f"targets must have shape [B], got {tuple(targets.shape)}.")
    if targets.dtype != torch.int64:
        raise TypeError("targets must be an int64 tensor.")
    if bool((targets < 0).any()) or bool((targets >= num_classes).any()):
        raise ValueError(f"targets must be in [0, {num_classes}).")


def _validate_prototype_inputs(
    batch: Batch,
    output_weak: ModelOutput,
    prototypes: Tensor | None,
) -> Tensor:
    if prototypes is None:
        raise ValueError("prototype signal requires prototypes shaped [C, D].")
    if output_weak.embedding.ndim != 2:
        raise ValueError(
            "output_weak.embedding must have shape [B, D], "
            f"got {tuple(output_weak.embedding.shape)}."
        )
    if not output_weak.embedding.is_floating_point():
        raise TypeError("output_weak.embedding must be floating-point.")
    if not torch.isfinite(output_weak.embedding).all():
        raise ValueError("output_weak.embedding contains NaN or Inf values.")
    if prototypes.ndim != 2 or prototypes.shape[1] != output_weak.embedding.shape[1]:
        raise ValueError(
            f"prototypes must have shape [C, D] matching embeddings, got {tuple(prototypes.shape)}."
        )
    if not prototypes.is_floating_point():
        raise TypeError("prototypes must be floating-point.")
    if not torch.isfinite(prototypes).all():
        raise ValueError("prototypes contains NaN or Inf values.")
    targets = batch.targets
    if targets is not None:
        _validate_targets(targets, output_weak.embedding.shape[0], prototypes.shape[0])
    return F.normalize(prototypes, dim=1)


def _require_finite_vector(values: Tensor, *, field_name: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{field_name} must return shape [B], got {tuple(values.shape)}.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{field_name} contains NaN or Inf values.")
