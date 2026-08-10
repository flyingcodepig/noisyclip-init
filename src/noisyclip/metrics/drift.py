"""Feature-drift metrics relative to frozen CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FeatureDriftMetrics:
    """Raw cosine and its legacy unit-interval alignment score."""

    cosine: float | None
    alignment: float | None
    reason: str | None = None


def feature_cosine_to_base(
    student_embedding: Tensor, base_embedding: Tensor | None
) -> FeatureDriftMetrics:
    """Compute raw mean cosine similarity to frozen CLIP features.

    Args:
        student_embedding: Floating tensor `[N, D]`.
        base_embedding: Optional floating tensor `[N, D]`.

    Returns:
        Raw mean cosine in `[-1, 1]` and the legacy mapped alignment score in
        `[0, 1]`; both are `None` when base embeddings are unavailable.

    Raises:
        ValueError: If shapes mismatch or values are non-finite.
    """

    if base_embedding is None:
        return FeatureDriftMetrics(None, None, "base CLIP embeddings unavailable")
    if student_embedding.shape != base_embedding.shape or student_embedding.ndim != 2:
        raise ValueError("student and base embeddings must have matching shape [N, D].")
    if student_embedding.shape[0] == 0:
        return FeatureDriftMetrics(None, None, "no samples")
    if not torch.isfinite(student_embedding).all() or not torch.isfinite(base_embedding).all():
        raise ValueError("embeddings contain NaN or Inf values.")
    cosine = F.cosine_similarity(student_embedding.float(), base_embedding.float(), dim=1).mean()
    raw = float(cosine.item())
    return FeatureDriftMetrics(raw, (raw + 1.0) / 2.0)
