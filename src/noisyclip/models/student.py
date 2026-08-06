"""Student model composed from a CLIP image backbone and classifier head."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from noisyclip.models.lora import lora_trainable_report, trainable_parameter_names
from noisyclip.models.outputs import ModelOutput
from noisyclip.models.validation import count_parameters, require_l2_normalized

StageName = Literal["B0", "B1", "B2"]


class NoisyCLIPStudent(nn.Module):
    """Backbone-plus-head student model.

    Args:
        backbone: Module exposing `embedding_dim` and `encode_image([B,3,224,224])`.
        head: Classifier module mapping `[B, D]` embeddings to `[B, C]` logits.
        stage: `B0` and `B1` freeze the full backbone; `B2` allows only LoRA
            adapter parameters in the backbone plus all head parameters.

    Raises:
        ValueError: If stage is unsupported, embeddings are not normalized, or
            unauthorized backbone parameters are trainable.
    """

    backbone: nn.Module
    head: nn.Module

    def __init__(self, *, backbone: nn.Module, head: nn.Module, stage: StageName = "B0") -> None:
        super().__init__()
        if stage not in ("B0", "B1", "B2"):
            raise ValueError(f"Unsupported student stage: {stage!r}.")
        if not callable(getattr(backbone, "encode_image", None)):
            raise AttributeError("backbone must expose encode_image(images).")
        if not hasattr(backbone, "embedding_dim"):
            raise AttributeError("backbone must expose embedding_dim.")
        self.backbone = backbone
        self.head = head
        self.stage = stage
        if stage in ("B0", "B1"):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self._validate_trainability()

    def forward(self, images: Tensor) -> ModelOutput:
        """Return model output for images shaped `[B, 3, 224, 224]`.

        Args:
            images: Floating-point finite image tensor `[B, 3, 224, 224]`.

        Returns:
            `ModelOutput` with logits `[B, C]`, L2-normalized embedding
            `[B, D]`, and optional scalar temperature.

        Raises:
            ValueError: If embedding normalization or logit batch size is
                invalid.
        """

        embedding = self.backbone.encode_image(images)
        require_l2_normalized(embedding)
        logits = self.head(embedding)
        if logits.ndim != 2 or logits.shape[0] != embedding.shape[0]:
            raise ValueError(
                "Classifier head must return logits shaped [B, C] with matching B, "
                f"got {tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Student logits contain NaN or Inf values.")
        temperature = None
        current_temperature = getattr(self.head, "current_temperature", None)
        if callable(current_temperature):
            temperature = current_temperature()
        return ModelOutput(logits=logits, embedding=embedding, temperature=temperature)

    def trainable_parameter_report(self) -> dict[str, int | float]:
        """Return total, trainable, head, LoRA, and unexpected parameter counts."""

        self._validate_trainability()
        total = count_parameters(self)
        trainable = count_parameters(self, trainable_only=True)
        head_trainable = count_parameters(self.head, trainable_only=True)
        lora_report = lora_trainable_report(self.backbone)
        unexpected = self._unexpected_backbone_trainables()
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_ratio": float(trainable / total) if total else 0.0,
            "head_trainable_parameters": head_trainable,
            "lora_trainable_parameters": lora_report.parameter_count,
            "lora_adapter_count": lora_report.adapter_count,
            "unexpected_trainable_parameters": sum(
                parameter.numel()
                for name, parameter in self.backbone.named_parameters()
                if name in unexpected
            ),
        }

    def export_single_model(
        self,
        destination: Path,
        *,
        preprocessing_spec: Mapping[str, Any] | None = None,
        config_summary: Mapping[str, Any] | None = None,
        class_to_idx: Mapping[str, int] | None = None,
        mapping_digest: str | None = None,
        clip_weight_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Merge LoRA and export one inference model artifact.

        Args:
            destination: Output `.pt` path. Parent directories may be created by
                the export helper.

        Returns:
            The written artifact path.
        """

        from noisyclip.models.export import export_student_model

        return export_student_model(
            self,
            destination,
            preprocessing_spec=preprocessing_spec,
            config_summary=config_summary,
            class_to_idx=class_to_idx,
            mapping_digest=mapping_digest,
            clip_weight_metadata=clip_weight_metadata,
        )

    def _validate_trainability(self) -> None:
        unexpected = self._unexpected_backbone_trainables()
        if unexpected:
            raise ValueError(f"Unauthorized trainable backbone parameters: {unexpected}.")

    def _unexpected_backbone_trainables(self) -> tuple[str, ...]:
        trainable = trainable_parameter_names(self.backbone)
        if self.stage in ("B0", "B1"):
            return trainable
        return tuple(name for name in trainable if ".lora_" not in f".{name}")
