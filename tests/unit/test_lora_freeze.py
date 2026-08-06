"""Tests for controlled LoRA injection, freezing, and merge equivalence."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from noisyclip.models.lora import (
    LoraInjectionConfig,
    LoRAMultiheadAttention,
    assert_only_lora_trainable,
    inject_lora_into_visual_transformer,
    merge_lora_adapters,
)


class TinyBlock(nn.Module):
    """Fake CLIP transformer block exposing fused attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=False)


class TinyVisual(nn.Module):
    """Fake CLIP visual tower with OpenAI-style transformer blocks."""

    def __init__(self, *, blocks: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = embed_dim
        self.stem = nn.Linear(3, embed_dim)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([TinyBlock(embed_dim) for _ in range(blocks)])
        self.proj = nn.Linear(embed_dim, embed_dim)


class TinyClip(nn.Module):
    """Fake CLIP model whose image path runs through attention blocks."""

    def __init__(self, *, blocks: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.visual = TinyVisual(blocks=blocks, embed_dim=embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized `[B,D]` features from `[B,3,224,224]` images."""

        x = self.visual.stem(images.mean(dim=(2, 3))).unsqueeze(0)
        for block in self.visual.transformer.resblocks:
            update, _ = block.attn(x, x, x, need_weights=False)
            x = x + update
        return F.normalize(self.visual.proj(x.squeeze(0)), dim=1)


def test_lora_injects_only_requested_blocks_and_projections() -> None:
    """Only configured block q/v adapters remain trainable."""

    model = TinyClip(blocks=4, embed_dim=8)
    report = inject_lora_into_visual_transformer(
        model,
        LoraInjectionConfig(target_blocks=(1, 3), target_projections=("q", "v"), rank=2, alpha=4),
    )

    assert report.adapter_count == 2
    assert report.parameter_count == 2 * 2 * (2 * 8 + 8 * 2)
    assert isinstance(model.visual.transformer.resblocks[1].attn, LoRAMultiheadAttention)
    assert isinstance(model.visual.transformer.resblocks[3].attn, LoRAMultiheadAttention)
    assert not isinstance(model.visual.transformer.resblocks[0].attn, LoRAMultiheadAttention)
    assert all(".lora_" in name for name in report.parameter_names)


def test_lora_default_targets_last_four_qv_blocks() -> None:
    """The default policy targets the final four visual blocks with q/v LoRA."""

    model = TinyClip(blocks=6, embed_dim=8)
    report = inject_lora_into_visual_transformer(model)

    assert report.adapter_count == 4
    assert not isinstance(model.visual.transformer.resblocks[1].attn, LoRAMultiheadAttention)
    assert isinstance(model.visual.transformer.resblocks[2].attn, LoRAMultiheadAttention)
    assert set(model.visual.transformer.resblocks[2].attn.target_projections) == {"q", "v"}


def test_lora_detects_unauthorized_trainable_backbone_parameter() -> None:
    """Backbone parameters outside LoRA adapters cannot remain trainable."""

    model = TinyClip(blocks=2, embed_dim=8)
    inject_lora_into_visual_transformer(
        model,
        LoraInjectionConfig(target_blocks=(0,), target_projections=("q",), rank=2, alpha=2),
    )
    model.visual.stem.weight.requires_grad = True

    with pytest.raises(ValueError, match="Unauthorized"):
        assert_only_lora_trainable(model)


def test_lora_merge_preserves_fp32_logits_and_is_idempotent_after_replacement() -> None:
    """Merged attention produces numerically equivalent logits and removes adapters."""

    torch.manual_seed(7)
    model = TinyClip(blocks=2, embed_dim=8).eval()
    inject_lora_into_visual_transformer(
        model,
        LoraInjectionConfig(target_blocks=(0, 1), target_projections=("q", "v"), rank=2, alpha=2),
    )
    for module in model.modules():
        if isinstance(module, LoRAMultiheadAttention):
            with torch.no_grad():
                for parameter in module.lora_b.parameters():
                    parameter.fill_(0.01)
    images = torch.randn(3, 3, 224, 224)

    with torch.inference_mode():
        before = model.encode_image(images)
    merged = merge_lora_adapters(model)
    with torch.inference_mode():
        after = model.encode_image(images)

    assert merged == 2
    assert merge_lora_adapters(model) == 0
    assert torch.max(torch.abs(before - after)).item() < 1e-5


def test_adapter_rejects_repeated_direct_merge() -> None:
    """Calling merge twice on the same wrapper fails clearly."""

    adapter = LoRAMultiheadAttention(
        nn.MultiheadAttention(8, num_heads=2),
        target_projections=("q",),
        rank=2,
        alpha=2,
        dropout=0.0,
    )

    adapter.merge()
    with pytest.raises(RuntimeError, match="already been merged"):
        adapter.merge()


@pytest.mark.parametrize(
    "config, error",
    [
        (LoraInjectionConfig(target_blocks=(99,), rank=2, alpha=2), IndexError),
        (
            LoraInjectionConfig(target_blocks=(0,), target_projections=("o",), rank=2, alpha=2),
            ValueError,
        ),
        (LoraInjectionConfig(target_blocks=(0,), rank=0, alpha=2), ValueError),
    ],
)
def test_lora_invalid_configuration_fails(
    config: LoraInjectionConfig,
    error: type[Exception],
) -> None:
    """Invalid blocks, projections, and rank fail before training."""

    with pytest.raises(error):
        inject_lora_into_visual_transformer(TinyClip(blocks=2), config)
