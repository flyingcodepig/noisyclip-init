"""Integration tests for single-model export equivalence."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import LinearClassifierHead
from noisyclip.models.export import export_student_model, load_exported_model
from noisyclip.models.lora import (
    LoraInjectionConfig,
    LoRAMultiheadAttention,
    inject_lora_into_visual_transformer,
)
from noisyclip.models.student import NoisyCLIPStudent


class TinyBlock(nn.Module):
    """Fake OpenAI-style block with fused q/k/v attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2)


class TinyClip(nn.Module):
    """Fake CLIP model that can be exported without real weights."""

    def __init__(self, *, blocks: int = 2, embed_dim: int = 8) -> None:
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
        """Return normalized image embeddings shaped `[B,D]`."""

        x = self.visual.stem(images.mean(dim=(2, 3))).unsqueeze(0)
        for block in self.visual.transformer.resblocks:
            update, _ = block.attn(x, x, x, need_weights=False)
            x = x + update
        return F.normalize(self.visual.proj(x.squeeze(0)), dim=1)


def _build_student(*, with_lora: bool) -> NoisyCLIPStudent:
    clip = TinyClip()
    if with_lora:
        inject_lora_into_visual_transformer(
            clip,
            LoraInjectionConfig(
                target_blocks=(0, 1), target_projections=("q", "v"), rank=2, alpha=2
            ),
        )
        for module in clip.modules():
            if isinstance(module, LoRAMultiheadAttention):
                with torch.no_grad():
                    for parameter in module.lora_b.parameters():
                        parameter.fill_(0.02)
    backbone = CLIPImageBackbone(clip, freeze=False)
    head = LinearClassifierHead(embedding_dim=8, num_classes=3)
    return NoisyCLIPStudent(backbone=backbone, head=head, stage="B2" if with_lora else "B0")


def test_export_merges_lora_and_reloaded_logits_are_equivalent(tmp_path: Path) -> None:
    """Exported packages reload as one model with no LoRA runtime keys."""

    torch.manual_seed(11)
    student = _build_student(with_lora=True).eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        before = student(images).logits.float()

    artifact = export_student_model(
        student,
        tmp_path / "student_export.pt",
        preprocessing_spec={"image_size": 224, "center_crop": 224},
        config_summary={"model": "fake-vit-b32", "teacher": False},
    )
    reloaded = load_exported_model(
        artifact,
        backbone=CLIPImageBackbone(TinyClip(), freeze=True),
        head=LinearClassifierHead(embedding_dim=8, num_classes=3),
    )
    with torch.inference_mode():
        after = reloaded(images).logits.float()

    package = torch.load(artifact, map_location="cpu")
    assert torch.max(torch.abs(before - after)).item() < 1e-5
    assert package["contains_teacher"] is False
    assert package["contains_optimizer"] is False
    assert package["contains_second_model"] is False
    assert package["num_classes"] == 3
    assert "weight_hash" in package
    assert all(".lora_" not in key for key in package["model_state"])


def test_export_rejects_non_student_shape(tmp_path: Path) -> None:
    """Export requires a module with one backbone and one head."""

    bad_model = nn.Linear(2, 2)

    try:
        export_student_model(bad_model, tmp_path / "bad.pt")
    except AttributeError as error:
        assert "backbone and head" in str(error)
    else:
        raise AssertionError("export_student_model should reject non-student modules")
