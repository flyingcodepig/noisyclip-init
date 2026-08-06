"""Frozen-teacher feature anchoring loss."""

from __future__ import annotations

import torch
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.losses._validation import (
    require_batch_alignment,
    require_model_output,
    require_normalized_embeddings,
    require_scalar,
)
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


class FeatureAnchorLoss:
    """Cosine anchor between student and frozen-teacher embeddings.

    Args:
        norm_tolerance: Absolute tolerance for validating L2-normalized rows.

    Inputs:
        Student embedding and teacher embedding must both be finite `[B, D]`
        floating tensors with row L2 norms close to `1`. The teacher embedding is
        detached before computing the target cosine.

    Outputs:
        A finite scalar mean of `1 - cosine(student, teacher)`, with gradient
        flowing only into the student embedding.

    Raises:
        ValueError: If teacher embedding is missing, shapes mismatch, embeddings
            are non-finite, or rows are not normalized.
    """

    name = "loss/feature_anchor"

    def __init__(self, *, norm_tolerance: float = 1e-4) -> None:
        if norm_tolerance <= 0.0:
            raise ValueError("norm_tolerance must be positive.")
        self.norm_tolerance = norm_tolerance

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
    ) -> Tensor:
        """Compute mean cosine anchor for embeddings `[B, D]`.

        Args:
            batch: Batch with unique sample IDs.
            student_weak: Student output carrying normalized embeddings `[B, D]`.
            teacher_embedding: Frozen teacher embeddings `[B, D]`.
            sample_states: Batch-aligned states used for ID checks.

        Returns:
            A finite scalar tensor.

        Raises:
            ValueError: If teacher embeddings are absent, mis-shaped,
                non-finite, or not L2-normalized.
        """

        batch_size, _ = require_model_output("student_weak", student_weak)
        require_batch_alignment(batch, batch_size, sample_states)
        if teacher_embedding is None:
            raise ValueError("teacher_embedding is required when feature anchor loss is enabled.")
        require_normalized_embeddings(
            "student_weak.embedding",
            student_weak.embedding,
            self.norm_tolerance,
        )
        require_normalized_embeddings("teacher_embedding", teacher_embedding, self.norm_tolerance)
        if teacher_embedding.shape != student_weak.embedding.shape:
            raise ValueError(
                "teacher_embedding must match student_weak.embedding shape "
                f"{tuple(student_weak.embedding.shape)}, got {tuple(teacher_embedding.shape)}."
            )

        teacher_target = teacher_embedding.detach()
        cosine = (student_weak.embedding * teacher_target).sum(dim=1)
        if not torch.isfinite(cosine).all():
            raise ValueError("feature anchor cosine values must be finite.")
        loss = (1.0 - cosine).mean()
        require_scalar(self.name, loss)
        return loss
