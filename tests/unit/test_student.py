"""Tests for the composed NoisyCLIP student model."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import LinearClassifierHead
from noisyclip.models.lora import LoraInjectionConfig, inject_lora_into_visual_transformer
from noisyclip.models.student import NoisyCLIPStudent


class TinyBlock(nn.Module):
    """Fake block with fused self-attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2)


class TinyClip(nn.Module):
    """Fake CLIP model used by student tests."""

    def __init__(self, *, blocks: int = 2, embed_dim: int = 6) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.output_dim = embed_dim
        self.visual.stem = nn.Linear(3, embed_dim)
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList(
            [TinyBlock(embed_dim) for _ in range(blocks)]
        )
        self.visual.proj = nn.Linear(embed_dim, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized `[B,D]` embeddings."""

        x = self.visual.stem(images.mean(dim=(2, 3))).unsqueeze(0)
        for block in self.visual.transformer.resblocks:
            update, _ = block.attn(x, x, x, need_weights=False)
            x = x + update
        return F.normalize(self.visual.proj(x.squeeze(0)), dim=1)


def test_student_b0_freezes_backbone_and_returns_model_output() -> None:
    """B0 students train the classifier head while embeddings stay normalized."""

    backbone = CLIPImageBackbone(TinyClip(), freeze=False)
    head = LinearClassifierHead(embedding_dim=6, num_classes=3)

    student = NoisyCLIPStudent(backbone=backbone, head=head, stage="B0")
    output = student(torch.randn(4, 3, 224, 224))
    report = student.trainable_parameter_report()

    assert output.logits.shape == (4, 3)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(4), atol=1e-5)
    assert report["head_trainable_parameters"] == report["trainable_parameters"]
    assert report["lora_trainable_parameters"] == 0
    assert all(not parameter.requires_grad for parameter in student.backbone.parameters())


def test_student_b2_allows_only_head_and_lora_backbone_parameters() -> None:
    """B2 students permit configured LoRA adapters plus classifier head."""

    clip = TinyClip(blocks=2, embed_dim=6)
    inject_lora_into_visual_transformer(
        clip,
        LoraInjectionConfig(target_blocks=(1,), target_projections=("q", "v"), rank=2, alpha=2),
    )
    backbone = CLIPImageBackbone(clip, freeze=False)
    head = LinearClassifierHead(embedding_dim=6, num_classes=3)

    student = NoisyCLIPStudent(backbone=backbone, head=head, stage="B2")
    report = student.trainable_parameter_report()

    assert report["head_trainable_parameters"] > 0
    assert report["lora_trainable_parameters"] > 0
    assert report["unexpected_trainable_parameters"] == 0


def test_student_rejects_unexpected_b2_backbone_trainable() -> None:
    """B2 validation fails when non-LoRA backbone weights are trainable."""

    backbone = CLIPImageBackbone(TinyClip(), freeze=False)
    head = LinearClassifierHead(embedding_dim=6, num_classes=3)

    with pytest.raises(ValueError, match="Unauthorized"):
        NoisyCLIPStudent(backbone=backbone, head=head, stage="B2")


def test_student_rejects_head_with_wrong_batch() -> None:
    """Classifier heads must return `[B,C]` logits for the same batch size."""

    class BadHead(nn.Module):
        num_classes = 3

        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """Return the wrong batch size."""

            return torch.zeros((embeddings.shape[0] + 1, 3))

    student = NoisyCLIPStudent(
        backbone=CLIPImageBackbone(TinyClip()),
        head=BadHead(),
        stage="B0",
    )

    with pytest.raises(ValueError, match="matching B"):
        student(torch.randn(2, 3, 224, 224))
