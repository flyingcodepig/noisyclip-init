"""Tests for frozen training-only teacher behavior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from noisyclip.models.teacher import FrozenTeacherModel


class TinyEncoder(nn.Module):
    """Small encoder exposing CLIP-like image embeddings."""

    embedding_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, self.embedding_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized `[B,4]` embeddings."""

        return F.normalize(self.proj(images.mean(dim=(2, 3))), dim=1)


def test_teacher_is_always_eval_and_frozen() -> None:
    """Construction and train calls keep the teacher eval-only and frozen."""

    teacher = FrozenTeacherModel(TinyEncoder(), embedding_dim=4)

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    teacher.train(True)

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_teacher_encode_uses_inference_mode() -> None:
    """Teacher outputs never require gradients even for grad-enabled inputs."""

    teacher = FrozenTeacherModel(TinyEncoder(), embedding_dim=4)
    images = torch.randn(2, 3, 224, 224, requires_grad=True)

    embedding = teacher.encode_image(images)

    assert embedding.shape == (2, 4)
    assert not embedding.requires_grad
    assert torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5)
