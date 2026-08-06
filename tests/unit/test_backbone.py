"""Tests for CLIP image backbone validation and freezing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from noisyclip.models.backbone import CLIPImageBackbone


class TinyClip(nn.Module):
    """Fake CLIP image model with a trainable projection."""

    def __init__(self, *, output_dim: int = 5, non_finite: bool = False) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.output_dim = output_dim
        self.proj = nn.Linear(3, output_dim)
        self.non_finite = non_finite

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Map `[B,3,224,224]` images to `[B,D]` features."""

        output = self.proj(images.mean(dim=(2, 3)))
        if self.non_finite:
            output = output.clone()
            output[0, 0] = float("nan")
        return output


def test_backbone_returns_l2_normalized_embeddings_and_freezes() -> None:
    """The wrapper validates input and returns finite normalized `[B,D]` output."""

    backbone = CLIPImageBackbone(TinyClip(), freeze=True)
    images = torch.randn(2, 3, 224, 224)

    embedding = backbone.encode_image(images)

    assert embedding.shape == (2, 5)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5)
    assert backbone.all_frozen()
    assert all(not flag for flag in backbone.requires_grad_report().values())


def test_backbone_can_report_unfrozen_parameters() -> None:
    """The freeze flag preserves actual requires_grad state when requested."""

    backbone = CLIPImageBackbone(TinyClip(), freeze=False)

    assert not backbone.all_frozen()
    assert any(backbone.requires_grad_report().values())


@pytest.mark.parametrize(
    "images, match",
    [
        (torch.randn(0, 3, 224, 224), "non-empty"),
        (torch.randn(2, 1, 224, 224), "shape"),
        (torch.ones(2, 3, 224, 224, dtype=torch.int64), "floating-point"),
    ],
)
def test_backbone_rejects_bad_inputs(images: torch.Tensor, match: str) -> None:
    """Input shape, dtype, and batch-size errors fail before model execution."""

    backbone = CLIPImageBackbone(TinyClip())

    with pytest.raises((TypeError, ValueError), match=match):
        backbone.encode_image(images)


def test_backbone_rejects_non_finite_encoder_output() -> None:
    """Non-finite CLIP features are never silently normalized."""

    backbone = CLIPImageBackbone(TinyClip(non_finite=True))

    with pytest.raises(ValueError, match="NaN or Inf"):
        backbone.encode_image(torch.randn(2, 3, 224, 224))
