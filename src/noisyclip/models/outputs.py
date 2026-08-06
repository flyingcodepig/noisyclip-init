"""Model outputs and model protocols from the shared architecture contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


@dataclass(slots=True)
class ModelOutput:
    """Output of a student image model.

    `logits` has shape `[B, C]` and is not softmax-normalized. `embedding` has
    shape `[B, D]` and must be L2-normalized by concrete implementations.
    `temperature` is either a scalar tensor or `None`.
    """

    logits: Tensor
    embedding: Tensor
    temperature: Tensor | None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


class ImageEncoder(Protocol):
    """Protocol for image backbones that return normalized embeddings only."""

    embedding_dim: int

    def encode_image(self, images: Tensor) -> Tensor:
        """Return `[B, D]` L2-normalized image features, never logits."""


class ClassifierHead(Protocol):
    """Protocol for classifier heads placed on top of image embeddings."""

    num_classes: int

    def __call__(self, embeddings: Tensor) -> Tensor:
        """Return unnormalized `[B, C]` logits from `[B, D]` embeddings."""


class StudentModel(Protocol):
    """Protocol for the trainable student model and its export surface."""

    def forward(self, images: Tensor) -> ModelOutput:
        """Return `ModelOutput` for images shaped `[B, 3, 224, 224]`."""

    def trainable_parameter_report(self) -> dict[str, int | float]:
        """Return counts and ratios for all trainable parameter groups."""

    def export_single_model(self, destination: Path) -> Path:
        """Export one inference model artifact and return its path."""


class FrozenTeacher(Protocol):
    """Protocol for frozen teacher encoders used during training only."""

    @torch.inference_mode()
    def encode_image(self, images: Tensor) -> Tensor:
        """Return `[B, D]` L2-normalized image features without gradients."""
