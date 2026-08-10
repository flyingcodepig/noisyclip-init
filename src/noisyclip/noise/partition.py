"""Class-wise trusted/uncertain/suspicious partitioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch
from torch import Tensor

from noisyclip.noise.normalize import percentile_rank_by_class
from noisyclip.noise.state import VALID_PARTITIONS, SampleState


def partition_by_class(
    sample_ids: Sequence[str],
    targets: Tensor,
    trust_scores: Tensor,
    *,
    trusted_quantile: float,
    uncertain_quantile: float,
    min_samples_per_class: int = 2,
) -> dict[str, str]:
    """Assign mutually exclusive partitions using class-wise trust ranks.

    Args:
        sample_ids: Stable IDs of length `N`; duplicates are rejected.
        targets: Int64 class tensor shaped `[N]` with values in `[0, C)`.
        trust_scores: Floating-point trust tensor shaped `[N]` in `[0, 1]`.
        trusted_quantile: Rank cutoff in `[0, 1]`; samples with rank greater
            than or equal to it are `trusted`.
        uncertain_quantile: Coverage cutoff in `[0, 1]`; samples with rank
            below `1 - uncertain_quantile` are `suspicious`, and the middle band
            is `uncertain`.
        min_samples_per_class: Classes smaller than this are deterministically
            marked `uncertain`.

    Returns:
        Mapping from each `sample_id` to one of `trusted`, `uncertain`, or
        `suspicious`; all samples are covered exactly once.

    Raises:
        TypeError: If tensor dtypes are invalid.
        ValueError: If shapes, ids, ranges, or quantiles are invalid.
    """

    ids = _validate_ids(sample_ids)
    _validate_partition_inputs(
        targets,
        trust_scores,
        len(ids),
        trusted_quantile=trusted_quantile,
        uncertain_quantile=uncertain_quantile,
        min_samples_per_class=min_samples_per_class,
    )
    num_classes = int(targets.max().item()) + 1
    ranks = percentile_rank_by_class(
        trust_scores,
        targets,
        num_classes,
        higher_is_better=True,
    )
    partitions: dict[str, str] = {}
    suspicious_cutoff = 1.0 - uncertain_quantile
    for class_index in range(num_classes):
        class_indices = torch.nonzero(targets == class_index, as_tuple=False).flatten()
        if class_indices.numel() == 0:
            continue
        if class_indices.numel() < min_samples_per_class:
            for index in class_indices.tolist():
                partitions[ids[index]] = "uncertain"
            continue
        ordered_indices = sorted(
            class_indices.tolist(),
            key=lambda index: (-float(ranks[index].item()), ids[index]),
        )
        for index in ordered_indices:
            rank = float(ranks[index].item())
            if rank >= trusted_quantile:
                partition = "trusted"
            elif rank < suspicious_cutoff:
                partition = "suspicious"
            else:
                partition = "uncertain"
            partitions[ids[index]] = partition
    if set(partitions) != set(ids):
        raise ValueError("partitioning failed to cover every sample_id exactly once.")
    return partitions


def apply_partitions(
    states: Sequence[SampleState],
    partitions: Mapping[str, str],
    *,
    epoch: int,
) -> list[SampleState]:
    """Return states updated with externally computed sample partitions.

    Args:
        states: Sequence of `SampleState` objects.
        partitions: Mapping from each state `sample_id` to a valid partition.
        epoch: Non-negative epoch used for `updated_epoch`.

    Returns:
        New `SampleState` objects with updated partition fields.

    Raises:
        ValueError: If ids are missing/extra, partitions are invalid, or epoch
            is negative.
    """

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}.")
    state_ids = [state.sample_id for state in states]
    _validate_ids(state_ids)
    missing = sorted(set(state_ids) - set(partitions))
    extra = sorted(set(partitions) - set(state_ids))
    if missing or extra:
        raise ValueError(f"partition ids must match state ids: missing={missing}, extra={extra}.")
    updated: list[SampleState] = []
    for state in states:
        partition = partitions[state.sample_id]
        if partition not in VALID_PARTITIONS:
            raise ValueError(f"Invalid partition for sample_id={state.sample_id}: {partition!r}.")
        updated.append(replace(state, partition=partition, updated_epoch=epoch))
    return updated


def apply_supervision_weights(
    states: Sequence[SampleState],
    *,
    trusted: float,
    uncertain_min: float,
    uncertain_max: float,
    suspicious: float,
    epoch: int,
) -> list[SampleState]:
    """Map partitions and trust scores to non-curricular base weights."""

    if not 0.0 <= suspicious <= uncertain_min <= uncertain_max <= trusted <= 1.0:
        raise ValueError(
            "supervision weights must satisfy "
            "0 <= suspicious <= uncertain_min <= uncertain_max <= trusted <= 1"
        )
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    updated: list[SampleState] = []
    for state in states:
        if state.partition == "trusted":
            weight = trusted
        elif state.partition == "suspicious":
            weight = suspicious
        elif state.partition == "uncertain":
            weight = uncertain_min + (uncertain_max - uncertain_min) * state.trust_score
        else:
            raise ValueError(
                f"Invalid partition for sample_id={state.sample_id}: {state.partition!r}."
            )
        updated.append(
            replace(state, supervised_weight=float(weight), updated_epoch=epoch)
        )
    return updated


def _validate_ids(sample_ids: Sequence[str]) -> list[str]:
    ids = list(sample_ids)
    if not ids:
        raise ValueError("sample_ids must be non-empty.")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("sample_ids must contain non-empty strings.")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicate sample_id values.")
    return ids


def _validate_partition_inputs(
    targets: Tensor,
    trust_scores: Tensor,
    expected_count: int,
    *,
    trusted_quantile: float,
    uncertain_quantile: float,
    min_samples_per_class: int,
) -> None:
    if not 0.0 <= trusted_quantile <= 1.0:
        raise ValueError("trusted_quantile must be in [0, 1].")
    if not 0.0 <= uncertain_quantile <= 1.0:
        raise ValueError("uncertain_quantile must be in [0, 1].")
    if trusted_quantile >= uncertain_quantile:
        raise ValueError("trusted_quantile must be lower than uncertain_quantile.")
    if min_samples_per_class < 1:
        raise ValueError("min_samples_per_class must be positive.")
    if targets.ndim != 1 or targets.shape[0] != expected_count:
        raise ValueError(f"targets must have shape [N], got {tuple(targets.shape)}.")
    if targets.dtype != torch.int64:
        raise TypeError("targets must be an int64 tensor.")
    if bool((targets < 0).any()):
        raise ValueError("targets must be non-negative.")
    if trust_scores.ndim != 1 or trust_scores.shape[0] != expected_count:
        raise ValueError(f"trust_scores must have shape [N], got {tuple(trust_scores.shape)}.")
    if not trust_scores.is_floating_point():
        raise TypeError("trust_scores must be a floating-point tensor.")
    if not torch.isfinite(trust_scores).all():
        raise ValueError("trust_scores contains NaN or Inf values.")
    if bool((trust_scores < 0).any()) or bool((trust_scores > 1).any()):
        raise ValueError("trust_scores must be in [0, 1].")
