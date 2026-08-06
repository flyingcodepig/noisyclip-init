"""Frozen teacher encoder used only for training-time feature constraints."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from noisyclip.models.validation import require_embedding_batch, require_image_batch


class FrozenTeacherModel(nn.Module):
    """Inference-mode wrapper around an image encoder.

    Args:
        encoder: Module implementing `encode_image(images) -> [B, D]`.
        embedding_dim: Expected embedding dimension `D`.

    Raises:
        ValueError: If encoded features are not finite `[B, D]`.
    """

    embedding_dim: int

    def __init__(self, encoder: nn.Module, *, embedding_dim: int) -> None:
        super().__init__()
        if not callable(getattr(encoder, "encode_image", None)):
            raise AttributeError("encoder must expose encode_image(images).")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def encode_image(self, images: Tensor) -> Tensor:
        """Return frozen `[B, D]` normalized teacher features.

        Args:
            images: Floating-point tensor shaped `[B, 3, 224, 224]`.

        Returns:
            Finite `[B, D]` embedding tensor produced under inference mode.

        Raises:
            TypeError: If images or features are not floating point.
            ValueError: If shape or finiteness checks fail.
        """

        require_image_batch(images)
        self.eval()
        embedding = self.encoder.encode_image(images)
        require_embedding_batch(
            embedding, embedding_dim=self.embedding_dim, field_name="teacher_embedding"
        )
        return embedding

    def train(self, mode: bool = True) -> FrozenTeacherModel:
        """Keep the teacher in eval mode even if callers request training mode."""

        super().train(False)
        return self
