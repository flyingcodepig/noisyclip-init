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
    """Measure top-1 prediction agreement over a fixed epoch window.

    Inputs use `output_weak.logits` shaped `[B, C]` and the per-sample top-1
    prediction history. Samples without a full window receive neutral score
    `0.5`; otherwise the score is the fraction of entries matching the current
    prediction. Higher values are more trustworthy.

    Raises:
        ValueError: If the window, prediction history, shapes, or values are invalid.
    """

    name: str = "prediction_stability"
    window: int = 3

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be positive.")

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
        predictions = output_weak.logits.argmax(dim=1).detach().cpu().tolist()
        values: list[float] = []
        for index, item in enumerate(state):
            values.append(
                prediction_stability_from_history(
                    item.prediction_history,
                    current=predictions[index],
                    window=self.window,
                )
            )
        result = output_weak.logits.new_tensor(values)
        _require_finite_vector(result, field_name=self.name)
        return result


def update_prediction_history(
    states: list[SampleState],
    logits: Tensor,
    *,
    epoch: int,
    momentum: float = 0.9,
    history_window: int = 3,
) -> list[SampleState]:
    """Update per-sample probability EMA and observation count once per epoch.

    Args:
        states: Unique sample states aligned with logits rows.
        logits: Finite model logits shaped `[N, C]`.
        epoch: Non-negative epoch recorded in returned states.
        momentum: Probability EMA coefficient in `[0, 1)`.
        history_window: Number of recent top-1 predictions to retain.

    Returns:
        New states preserving input order, with normalized `ema_probs`,
        bounded top-1 prediction history, `seen_count + 1`, and
        `updated_epoch=epoch`.

    Raises:
        ValueError: If IDs, shapes, epoch, momentum, previous probability
            vectors, or logits are invalid.
    """

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    if history_window < 1:
        raise ValueError("history_window must be positive.")
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
        prior_predictions = (
            state.prediction_history[-(history_window - 1) :] if history_window > 1 else []
        )
        prediction_history = [*prior_predictions, int(current.argmax().item())]
        updated.append(
            replace(
                state,
                seen_count=state.seen_count + 1,
                ema_probs=history.tolist(),
                prediction_history=prediction_history,
                updated_epoch=epoch,
            )
        )
    return updated


def update_prediction_history_from_top1(
    states: list[SampleState],
    predictions: Tensor,
    *,
    epoch: int,
    history_window: int = 3,
) -> list[SampleState]:
    """Persist compact top-1 history without per-class probability vectors."""

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}.")
    if history_window < 1:
        raise ValueError("history_window must be positive.")
    if predictions.shape != (len(states),) or predictions.dtype != torch.int64:
        raise ValueError(f"predictions must be int64 shape [{len(states)}].")
    if bool((predictions < 0).any()):
        raise ValueError("predictions must be non-negative.")
    sample_ids = [state.sample_id for state in states]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("states must contain unique, non-empty sample_id values.")

    updated: list[SampleState] = []
    for state, prediction in zip(states, predictions.tolist(), strict=True):
        prior = (
            state.prediction_history[-(history_window - 1) :] if history_window > 1 else []
        )
        updated.append(
            replace(
                state,
                seen_count=state.seen_count + 1,
                ema_probs=None,
                prediction_history=[*prior, int(prediction)],
                updated_epoch=epoch,
            )
        )
    return updated


def prediction_stability_from_history(
    history: list[int],
    *,
    current: int,
    window: int,
) -> float:
    """Return fixed-window top-1 agreement, or neutral score before warmup."""

    if window < 1:
        raise ValueError("window must be positive.")
    if current < 0 or any(value < 0 for value in history):
        raise ValueError("prediction indices must be non-negative.")
    prior = history[-(window - 1) :] if window > 1 else []
    values = [*prior, current]
    if len(values) < window:
        return 0.5
    return sum(value == current for value in values) / window


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
