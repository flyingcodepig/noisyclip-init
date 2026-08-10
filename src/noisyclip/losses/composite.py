"""Configured robust loss composition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from noisyclip.config.schema import LossConfig
from noisyclip.data.records import Batch
from noisyclip.losses._validation import (
    clone_component_mapping,
    require_batch_alignment,
    require_model_output,
    require_scalar,
    supervised_weights,
)
from noisyclip.losses.consistency import ConsistencyLoss
from noisyclip.losses.elr import ELRLoss
from noisyclip.losses.feature_anchor import FeatureAnchorLoss
from noisyclip.losses.outputs import LossOutput
from noisyclip.losses.weighted_ce import WeightedCrossEntropyLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


class RobustCompositeLoss:
    """Assemble enabled robust loss terms from `LossConfig`.

    Args:
        config: Existing immutable loss configuration. The composite preserves
            public fields and uses `enabled` plus `weight` to decide which
            components contribute.

    Inputs:
        `student_weak.logits` must be finite `[B, C]`, embeddings must be finite
        `[B, D]`, `sample_states` must match unique `batch.sample_ids`, and
        active terms may require targets, strong outputs, or teacher embeddings.

    Outputs:
        `LossOutput` with finite scalar `total`, stable component names
        (`loss/ce`, `loss/elr`, `loss/consistency`, `loss/feature_anchor`,
        `loss/effective_supervised_weight`), and detached `[B]`
        `per_sample_supervised` when CE is enabled.

    Raises:
        ValueError: If any enabled component receives invalid inputs or produces
            NaN/Inf.
    """

    def __init__(
        self,
        config: LossConfig,
        *,
        sample_ids: Sequence[str] | None = None,
        history_device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.ce = WeightedCrossEntropyLoss(
            label_smoothing=config.cross_entropy.label_smoothing,
        )
        self.elr = ELRLoss(
            target_momentum=config.elr.target_momentum,
            start_epoch=config.elr.start_epoch,
            enabled=config.elr.enabled,
            sample_ids=sample_ids,
            history_device=history_device,
        )
        self.consistency = ConsistencyLoss(
            temperature=config.consistency.temperature,
            start_epoch=config.consistency.start_epoch,
            enabled=config.consistency.enabled,
        )
        self.feature_anchor = FeatureAnchorLoss()

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> LossOutput:
        """Return configured total loss and finite scalar component mapping.

        Args:
            batch: Batch with unique `sample_ids`; active CE requires int64
                targets `[B]` in `[0, C)`.
            student_weak: Weak-view student output with logits `[B, C]`.
            student_strong: Strong-view student output required by enabled
                consistency loss.
            teacher_embedding: Frozen teacher embeddings `[B, D]` required by
                enabled feature anchoring.
            sample_states: Batch-aligned per-sample state carrying supervised
                weights in `[0, 1]`.
            epoch: Non-negative epoch index for warmup-gated terms.

        Returns:
            A `LossOutput` whose total and component tensors are finite scalars.

        Raises:
            ValueError: If the batch is malformed, an active component is
                impossible to compute, or any scalar is NaN/Inf.
        """

        batch_size, _ = require_model_output("student_weak", student_weak)
        require_batch_alignment(batch, batch_size, sample_states)
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        components: dict[str, Tensor] = {}
        total = student_weak.logits.new_zeros(())
        per_sample_supervised: Tensor | None = None
        effective_weight = supervised_weights(
            sample_states,
            device=student_weak.logits.device,
            dtype=student_weak.logits.dtype,
            require_positive=False,
        ).sum()
        components["loss/effective_supervised_weight"] = effective_weight

        if self.config.cross_entropy.enabled:
            ce_loss, per_sample = self.ce(batch, student_weak, sample_states)
            ce_component = ce_loss * self.config.cross_entropy.weight
            require_scalar("loss/ce", ce_component)
            components["loss/ce"] = ce_component
            total = total + ce_component
            per_sample_supervised = per_sample.detach()

        if self.config.elr.enabled:
            elr_loss = self.elr(batch, student_weak, sample_states, epoch)
            elr_component = elr_loss * self.config.elr.weight
            require_scalar("loss/elr", elr_component)
            components["loss/elr"] = elr_component
            total = total + elr_component

        if self.config.consistency.enabled:
            consistency_loss = self.consistency(
                batch,
                student_weak,
                student_strong,
                sample_states,
                epoch,
            )
            consistency_component = consistency_loss * self.config.consistency.weight
            require_scalar("loss/consistency", consistency_component)
            components["loss/consistency"] = consistency_component
            total = total + consistency_component

        if self.config.feature_anchor.enabled:
            anchor_loss = self.feature_anchor(
                batch,
                student_weak,
                teacher_embedding,
                sample_states,
            )
            anchor_component = anchor_loss * self.config.feature_anchor.weight
            require_scalar("loss/feature_anchor", anchor_component)
            components["loss/feature_anchor"] = anchor_component
            total = total + anchor_component

        require_scalar("loss/total", total)
        return LossOutput(
            total=total,
            components=clone_component_mapping(components),
            per_sample_supervised=per_sample_supervised,
        )
