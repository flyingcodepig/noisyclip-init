"""Class-wise trust aggregation and continuous supervised weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from noisyclip.data.records import SampleRecord
from noisyclip.noise.normalize import percentile_rank_by_class
from noisyclip.noise.state import SampleState

SIGNAL_HIGHER_IS_BETTER = {
    "ema_loss": False,
    "augmentation_agreement": True,
    "prototype_similarity": True,
    "prototype_margin": True,
    "prediction_stability": True,
}


@dataclass(frozen=True, slots=True)
class ClasswiseTrustAggregator:
    """Aggregate enabled raw signals into sample trust scores.

    Args:
        signal_coefficients: Mapping from signal names to non-negative
            coefficients. Positive entries are enabled and must be present in
            `raw_signals`.
        supervised_weight_min: Lower bound for continuous weights.
        supervised_weight_max: Upper bound for continuous weights.

    `records` are ordered samples with train/validation `target` values in
    `[0, C)`. Each raw signal is a `[N]` tensor. The output list preserves
    `records` order, stores class-wise normalized signal fields, and sets
    `trust_score` plus `supervised_weight` to finite values in `[0, 1]`.

    Raises:
        ValueError: If coefficients sum to zero, ids do not match, targets are
            missing, tensors have invalid shapes, or outputs are non-finite.
    """

    signal_coefficients: Mapping[str, float]
    supervised_weight_min: float = 0.0
    supervised_weight_max: float = 1.0

    def __post_init__(self) -> None:
        """Validate coefficient and supervised-weight ranges."""

        enabled_sum = sum(value for value in self.signal_coefficients.values() if value > 0.0)
        if enabled_sum <= 0.0:
            raise ValueError("At least one enabled trust signal coefficient must be positive.")
        for name, coefficient in self.signal_coefficients.items():
            if name not in SIGNAL_HIGHER_IS_BETTER:
                raise ValueError(f"Unsupported trust signal name: {name!r}.")
            if coefficient < 0.0:
                raise ValueError(f"Trust signal coefficient must be non-negative: {name}.")
        if not 0.0 <= self.supervised_weight_min <= self.supervised_weight_max <= 1.0:
            raise ValueError("supervised weight bounds must satisfy 0 <= min <= max <= 1.")

    @classmethod
    def from_config(cls, noise_config: Any) -> ClasswiseTrustAggregator:
        """Create an aggregator from a validated `NoiseConfig`-like object.

        Args:
            noise_config: Object with `signals` and `weights` attributes from
                the project configuration schema.

        Returns:
            A `ClasswiseTrustAggregator` honoring enabled signal coefficients.

        Raises:
            AttributeError: If expected config fields are absent.
            ValueError: If all enabled coefficients are zero.
        """

        signals = noise_config.signals
        coefficients = {
            "ema_loss": signals.ema_loss.coefficient if signals.ema_loss.enabled else 0.0,
            "augmentation_agreement": (
                signals.augmentation_agreement.coefficient
                if signals.augmentation_agreement.enabled
                else 0.0
            ),
            "prototype_similarity": (
                signals.prototype_similarity.coefficient
                if signals.prototype_similarity.enabled
                else 0.0
            ),
            "prototype_margin": (
                signals.prototype_margin.coefficient if signals.prototype_margin.enabled else 0.0
            ),
            "prediction_stability": (
                signals.prediction_stability.coefficient
                if signals.prediction_stability.enabled
                else 0.0
            ),
        }
        return cls(
            coefficients,
            supervised_weight_min=noise_config.weights.suspicious,
            supervised_weight_max=noise_config.weights.trusted,
        )

    def update_epoch(
        self,
        records: list[SampleRecord],
        raw_signals: Mapping[str, Tensor],
        previous: list[SampleState],
        epoch: int,
    ) -> list[SampleState]:
        """Return updated states after class-wise signal normalization."""

        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        if not records:
            raise ValueError("records must be non-empty.")
        record_ids = [record.sample_id for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("records contains duplicate sample_id values.")
        previous_by_id = {state.sample_id: state for state in previous}
        if len(previous_by_id) != len(previous):
            raise ValueError("previous contains duplicate sample_id values.")
        missing_previous = [
            sample_id for sample_id in record_ids if sample_id not in previous_by_id
        ]
        if missing_previous:
            raise ValueError(f"previous is missing sample_id(s): {missing_previous}.")
        targets = _targets_from_records(records)
        num_classes = int(targets.max().item()) + 1
        normalized_signals: dict[str, Tensor] = {}
        validated_raw_signals: dict[str, Tensor] = {}
        weighted_sum = torch.zeros(len(records), dtype=torch.float32)
        coefficient_sum = 0.0
        for name, coefficient in self.signal_coefficients.items():
            if coefficient <= 0.0:
                continue
            if name not in raw_signals:
                raise ValueError(f"raw_signals is missing enabled signal {name!r}.")
            raw = raw_signals[name].detach().to(torch.float32)
            _validate_raw_signal(raw, expected_count=len(records), name=name)
            validated_raw_signals[name] = raw
            normalized = percentile_rank_by_class(
                raw,
                targets,
                num_classes,
                higher_is_better=SIGNAL_HIGHER_IS_BETTER[name],
            )
            normalized_signals[name] = normalized
            weighted_sum = weighted_sum + normalized * float(coefficient)
            coefficient_sum += float(coefficient)
        if coefficient_sum <= 0.0:
            raise ValueError("Enabled trust signal coefficients sum to zero.")
        trust_score = (weighted_sum / coefficient_sum).clamp(0.0, 1.0)
        supervised_weight = (
            self.supervised_weight_min
            + (self.supervised_weight_max - self.supervised_weight_min) * trust_score
        ).clamp(0.0, 1.0)
        if not torch.isfinite(trust_score).all() or not torch.isfinite(supervised_weight).all():
            raise ValueError("trust_score and supervised_weight must be finite.")
        updated: list[SampleState] = []
        for index, record in enumerate(records):
            prior = previous_by_id[record.sample_id]
            updated.append(
                replace(
                    prior,
                    ema_loss=float(
                        validated_raw_signals.get("ema_loss", torch.tensor([prior.ema_loss]))[
                            index if "ema_loss" in normalized_signals else 0
                        ].item()
                    ),
                    prediction_stability=float(
                        normalized_signals.get(
                            "prediction_stability",
                            torch.tensor([prior.prediction_stability]),
                        )[index if "prediction_stability" in normalized_signals else 0].item()
                    ),
                    augmentation_agreement=float(
                        normalized_signals.get(
                            "augmentation_agreement",
                            torch.tensor([prior.augmentation_agreement]),
                        )[index if "augmentation_agreement" in normalized_signals else 0].item()
                    ),
                    prototype_similarity=float(
                        normalized_signals.get(
                            "prototype_similarity",
                            torch.tensor([prior.prototype_similarity]),
                        )[index if "prototype_similarity" in normalized_signals else 0].item()
                    ),
                    prototype_margin=float(
                        normalized_signals.get(
                            "prototype_margin",
                            torch.tensor([prior.prototype_margin]),
                        )[index if "prototype_margin" in normalized_signals else 0].item()
                    ),
                    trust_score=float(trust_score[index].item()),
                    supervised_weight=float(supervised_weight[index].item()),
                    updated_epoch=epoch,
                )
            )
        return updated


def _targets_from_records(records: list[SampleRecord]) -> Tensor:
    targets: list[int] = []
    for record in records:
        if record.target is None:
            raise ValueError(f"record target is missing for sample_id={record.sample_id}.")
        if record.target < 0:
            raise ValueError(
                f"record target must be non-negative for sample_id={record.sample_id}."
            )
        targets.append(record.target)
    return torch.tensor(targets, dtype=torch.int64)


def _validate_raw_signal(raw: Tensor, *, expected_count: int, name: str) -> None:
    if raw.ndim != 1 or raw.shape[0] != expected_count:
        raise ValueError(f"raw signal {name!r} must have shape [N], got {tuple(raw.shape)}.")
    if torch.isinf(raw).any():
        raise ValueError(f"raw signal {name!r} contains Inf values.")
    if name == "ema_loss" and torch.isnan(raw).any():
        raise ValueError("raw signal 'ema_loss' contains NaN values.")
