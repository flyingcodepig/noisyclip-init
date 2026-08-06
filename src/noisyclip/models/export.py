"""Single-model export and reload helpers for NoisyCLIP students."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from noisyclip.models.lora import has_lora_adapters, merge_lora_adapters
from noisyclip.models.outputs import ModelOutput

EXPORT_FORMAT_VERSION = 1


class ExportedInferenceModel(nn.Module):
    """Reloaded inference-only model composed from one backbone and one head.

    Args:
        backbone: Module exposing `encode_image([B,3,224,224]) -> [B,D]`.
        head: Module mapping `[B,D]` to `[B,C]` logits.

    Raises:
        AttributeError: If the backbone does not expose `encode_image`.
    """

    def __init__(self, *, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        if not callable(getattr(backbone, "encode_image", None)):
            raise AttributeError("backbone must expose encode_image(images).")
        self.backbone = backbone
        self.head = head
        self.eval()

    @torch.inference_mode()
    def forward(self, images: Tensor) -> ModelOutput:
        """Return logits and embeddings for one `[B,3,224,224]` image batch."""

        embedding = self.backbone.encode_image(images)
        logits = self.head(embedding)
        temperature = None
        current_temperature = getattr(self.head, "current_temperature", None)
        if callable(current_temperature):
            temperature = current_temperature()
        return ModelOutput(logits=logits, embedding=embedding, temperature=temperature)


def export_student_model(
    student: nn.Module,
    destination: Path | str,
    *,
    preprocessing_spec: Mapping[str, Any] | None = None,
    config_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Merge LoRA and write one reloadable inference artifact.

    Args:
        student: Student module with `backbone`, `head`, and `state_dict`.
        destination: Output `.pt` artifact path.
        preprocessing_spec: Optional immutable preprocessing summary, for
            example image size, resize, crop, mean, and std.
        config_summary: Optional sanitized model configuration summary.

    Returns:
        The written artifact path.

    Raises:
        AttributeError: If `student` lacks required backbone/head attributes.
        RuntimeError: If unmerged LoRA adapters remain after merge.
    """

    if not hasattr(student, "backbone") or not hasattr(student, "head"):
        raise AttributeError("student must expose backbone and head attributes.")
    student.eval()
    backbone = student.backbone
    merge_lora_adapters(backbone)
    if has_lora_adapters(student):
        raise RuntimeError("Export cannot contain unmerged LoRA runtime adapters.")

    state_dict = student.state_dict()
    state_hash = _hash_state_dict(state_dict)
    head = student.head
    package = {
        "format_version": EXPORT_FORMAT_VERSION,
        "model_state": state_dict,
        "structure": {
            "student_class": type(student).__name__,
            "backbone_class": type(backbone).__name__,
            "head_class": type(head).__name__,
            "embedding_dim": int(getattr(backbone, "embedding_dim", 0)),
        },
        "num_classes": int(getattr(head, "num_classes", 0)),
        "preprocessing": dict(preprocessing_spec or _default_preprocessing_spec()),
        "config_summary": dict(config_summary or {}),
        "weight_hash": {"algorithm": "sha256", "model_state": state_hash},
        "contains_teacher": False,
        "contains_optimizer": False,
        "contains_second_model": False,
    }
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output_path)
    return output_path


def load_exported_model(
    artifact_path: Path | str,
    *,
    backbone: nn.Module,
    head: nn.Module,
    map_location: str | torch.device = "cpu",
) -> ExportedInferenceModel:
    """Load an exported single-model package into supplied architecture modules.

    Args:
        artifact_path: Path written by `export_student_model`.
        backbone: Fresh backbone module matching the exported state.
        head: Fresh head module matching the exported state.
        map_location: Torch load map location.

    Returns:
        `ExportedInferenceModel` in eval mode.

    Raises:
        ValueError: If the artifact is not a NoisyCLIP single-model export.
        RuntimeError: If state loading fails.
    """

    package = torch.load(Path(artifact_path), map_location=map_location)
    if not isinstance(package, Mapping):
        raise ValueError("Export artifact must contain a mapping package.")
    if package.get("format_version") != EXPORT_FORMAT_VERSION:
        raise ValueError("Unsupported export format version.")
    if package.get("contains_teacher") is not False:
        raise ValueError("Export artifact must not contain a teacher model.")
    if package.get("contains_optimizer") is not False:
        raise ValueError("Export artifact must not contain optimizer state.")
    model = ExportedInferenceModel(backbone=backbone, head=head)
    model.load_state_dict(package["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def assert_export_equivalent(
    student: nn.Module,
    images: Tensor,
    artifact_path: Path | str,
    *,
    reloaded_model: nn.Module,
    atol: float = 1e-5,
) -> None:
    """Assert exported and reloaded FP32 logits are equivalent.

    Args:
        student: Source student model.
        images: Fixed floating-point input shaped `[B,3,224,224]`.
        artifact_path: Destination used for export.
        reloaded_model: Fresh model architecture loaded by caller after export.
        atol: Maximum absolute tolerance; default `1e-5`.

    Raises:
        AssertionError: If logits differ by more than `atol`.
    """

    student.eval()
    with torch.inference_mode():
        before = student(images).logits.float()
    export_student_model(student, artifact_path)
    with torch.inference_mode():
        after = reloaded_model(images).logits.float()
    max_abs = (before - after).abs().max().item()
    if max_abs > atol:
        raise AssertionError(f"Export logits differ by {max_abs:.6g}, tolerance is {atol}.")


def _hash_state_dict(state_dict: Mapping[str, Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _default_preprocessing_spec() -> dict[str, Any]:
    return {
        "image_size": 224,
        "resize_short_side": 256,
        "center_crop": 224,
        "input_shape": [3, 224, 224],
        "dtype": "float32",
        "normalization": "openai_clip_official",
    }
