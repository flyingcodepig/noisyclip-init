"""Feature-drift metrics relative to frozen CLIP embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from noisyclip.metrics.classification import MetricValue


def feature_cosine_to_base(student_embedding: Tensor, base_embedding: Tensor | None) -> MetricValue:
    """Compute mean cosine similarity to frozen CLIP features.

    Args:
        student_embedding: Floating tensor `[N, D]`.
        base_embedding: Optional floating tensor `[N, D]`.

    Returns:
        Mean cosine mapped from `[-1, 1]` into `[0, 1]`; `None` when base
        embeddings are unavailable.

    Raises:
        ValueError: If shapes mismatch or values are non-finite.
    """

    if base_embedding is None:
        return MetricValue(None, "base CLIP embeddings unavailable")
    if student_embedding.shape != base_embedding.shape or student_embedding.ndim != 2:
        raise ValueError("student and base embeddings must have matching shape [N, D].")
    if student_embedding.shape[0] == 0:
        return MetricValue(None, "no samples")
    if not torch.isfinite(student_embedding).all() or not torch.isfinite(base_embedding).all():
        raise ValueError("embeddings contain NaN or Inf values.")
    cosine = F.cosine_similarity(student_embedding.float(), base_embedding.float(), dim=1).mean()
    return MetricValue(float((cosine.item() + 1.0) / 2.0))
