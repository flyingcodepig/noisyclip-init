"""Single-model export and reload helpers for NoisyCLIP students."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import (
    CosineClassifierHead,
    LinearClassifierHead,
    build_classifier_head,
)
from noisyclip.models.clip_loader import ClipBackend, load_clip_vit_b32
from noisyclip.models.lora import has_lora_adapters, merge_lora_adapters
from noisyclip.models.outputs import ModelOutput

EXPORT_FORMAT_VERSION = 2
CLASS_ID_PATTERN = re.compile(r"^[0-9]{4}$")


class ExportFormatError(ValueError):
    """Raised when a model artifact violates the single-model export contract."""


class ExportedInferenceModel(nn.Module):
    """Reloaded inference-only model composed from one backbone and one head."""

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
    class_to_idx: Mapping[str, int] | None = None,
    mapping_digest: str | None = None,
    clip_weight_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Merge LoRA and write one auditable, reloadable inference artifact.

    The artifact contains exactly one student state. A class mapping is optional
    for model-only tests, but is required by the submission inference pipeline.
    Existing destinations are never overwritten.
    """

    if not hasattr(student, "backbone") or not hasattr(student, "head"):
        raise AttributeError("student must expose backbone and head attributes.")
    output_path = Path(destination)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite exported model: {output_path}")

    student.eval()
    backbone = student.backbone
    if has_lora_adapters(student):
        merge_lora_adapters(backbone)
    if has_lora_adapters(student):
        raise RuntimeError("Export cannot contain unmerged LoRA runtime adapters.")

    state_dict = {key: value.detach().cpu() for key, value in student.state_dict().items()}
    head_spec = _head_spec(student.head)
    structure = {
        "student_class": type(student).__name__,
        "backbone_class": type(backbone).__name__,
        "head_class": type(student.head).__name__,
        "embedding_dim": int(getattr(backbone, "embedding_dim", 0)),
        "backbone": {
            "name": "ViT-B/32",
            "pretrained": "openai",
            "embedding_dim": int(getattr(backbone, "embedding_dim", 0)),
        },
        "head": head_spec,
    }
    package: dict[str, Any] = {
        "artifact_type": "noisyclip_single_model_export",
        "format_version": EXPORT_FORMAT_VERSION,
        "models": [{"role": "student"}],
        "model_state": state_dict,
        "structure": structure,
        "num_classes": int(getattr(student.head, "num_classes", 0)),
        "preprocess": dict(preprocessing_spec or _default_preprocessing_spec()),
        "config_summary": dict(config_summary or {}),
        "clip_weight_metadata": dict(clip_weight_metadata or {}),
        "weight_hash": {"algorithm": "sha256", "model_state": _hash_state_dict(state_dict)},
        "contains_teacher": False,
        "contains_optimizer": False,
        "contains_second_model": False,
    }
    if class_to_idx is not None:
        normalized_mapping = _validate_class_mapping(class_to_idx, package["num_classes"])
        computed_digest = _mapping_digest(normalized_mapping)
        if mapping_digest is not None and mapping_digest != computed_digest:
            raise ExportFormatError(
                f"mapping_digest mismatch: expected {mapping_digest}, computed {computed_digest}."
            )
        package["class_to_idx"] = normalized_mapping
        package["mapping_digest"] = computed_digest
    elif mapping_digest is not None:
        raise ExportFormatError("mapping_digest cannot be supplied without class_to_idx.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary export: {temporary}")
    try:
        torch.save(package, temporary)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def load_export_package(artifact_path: Path | str) -> Mapping[str, Any]:
    """Load and validate one tensor-only single-student export package."""

    path = Path(artifact_path)
    try:
        package = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch.
        package = torch.load(path, map_location="cpu")
    if not isinstance(package, Mapping):
        raise ExportFormatError("Export artifact must contain a mapping package.")
    if package.get("artifact_type") != "noisyclip_single_model_export":
        raise ExportFormatError("Unsupported model artifact_type.")
    if package.get("format_version") != EXPORT_FORMAT_VERSION:
        raise ExportFormatError("Unsupported export format version.")
    models = package.get("models")
    if models != [{"role": "student"}]:
        raise ExportFormatError("Export must describe exactly one student model.")
    for key in ("contains_teacher", "contains_optimizer", "contains_second_model"):
        if package.get(key) is not False:
            raise ExportFormatError(f"Export package has forbidden or missing flag: {key}.")
    state = package.get("model_state")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ExportFormatError("model_state must be a string-to-tensor mapping.")
    weight_hash = package.get("weight_hash")
    if not isinstance(weight_hash, Mapping) or weight_hash.get("algorithm") != "sha256":
        raise ExportFormatError("Export package must contain a SHA256 model_state hash.")
    expected_hash = weight_hash.get("model_state")
    if not isinstance(expected_hash, str) or expected_hash != _hash_state_dict(state):
        raise ExportFormatError("Exported model_state SHA256 mismatch.")
    return package


def load_exported_model(
    artifact_path: Path | str,
    *,
    backbone: nn.Module,
    head: nn.Module,
    map_location: str | torch.device = "cpu",
) -> ExportedInferenceModel:
    """Load an exported package into caller-supplied architecture modules."""

    package = load_export_package(artifact_path)
    model = ExportedInferenceModel(backbone=backbone, head=head)
    model.load_state_dict(package["model_state"])
    model.to(map_location)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_exported_model_auto(
    artifact_path: Path | str,
    *,
    device: str | torch.device = "cpu",
    cache_dir: Path | str | None = None,
    backend: ClipBackend | None = None,
) -> ExportedInferenceModel:
    """Rebuild official CLIP ViT-B/32 and its head from export metadata."""

    package = load_export_package(artifact_path)
    structure = package.get("structure")
    if not isinstance(structure, Mapping):
        raise ExportFormatError("Export structure metadata is missing.")
    backbone_spec = structure.get("backbone")
    head_spec = structure.get("head")
    if not isinstance(backbone_spec, Mapping) or not isinstance(head_spec, Mapping):
        raise ExportFormatError("Export backbone/head structure metadata is invalid.")
    loaded = load_clip_vit_b32(
        model_name=str(backbone_spec.get("name")),
        pretrained=str(backbone_spec.get("pretrained")),
        device=device,
        cache_dir=cache_dir,
        backend=backend,
    )
    embedding_dim = _required_int(backbone_spec, "embedding_dim")
    backbone = CLIPImageBackbone(loaded.model, embedding_dim=embedding_dim, freeze=True)
    head = build_classifier_head(
        head_type=str(head_spec.get("type")),
        embedding_dim=embedding_dim,
        num_classes=_required_int(head_spec, "num_classes"),
        temperature_init=_optional_float(head_spec, "temperature_init"),
        temperature_min=_optional_float(head_spec, "temperature_min"),
        temperature_max=_optional_float(head_spec, "temperature_max"),
    )
    return load_exported_model(
        artifact_path,
        backbone=backbone,
        head=head,
        map_location=device,
    )


def assert_export_equivalent(
    student: nn.Module,
    images: Tensor,
    artifact_path: Path | str,
    *,
    reloaded_model: nn.Module,
    atol: float = 1e-5,
) -> None:
    """Assert source and already reloaded FP32 model logits are equivalent."""

    student.eval()
    with torch.inference_mode():
        before = student(images).logits.float()
        after = reloaded_model(images).logits.float()
    max_abs = (before - after).abs().max().item()
    if max_abs > atol:
        raise AssertionError(f"Export logits differ by {max_abs:.6g}, tolerance is {atol}.")


def _head_spec(head: nn.Module) -> dict[str, Any]:
    if isinstance(head, LinearClassifierHead):
        return {"type": "linear", "num_classes": head.num_classes}
    if isinstance(head, CosineClassifierHead):
        return {
            "type": "cosine",
            "num_classes": head.num_classes,
            "temperature_init": float(head.current_temperature().detach().cpu().item()),
            "temperature_min": head.temperature_min,
            "temperature_max": head.temperature_max,
        }
    raise ExportFormatError(f"Unsupported classifier head for export: {type(head).__name__}.")


def _hash_state_dict(state_dict: Mapping[str, Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: state_dict[key].detach().cpu() for key in sorted(state_dict)}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _mapping_digest(mapping: Mapping[str, int]) -> str:
    payload = json.dumps(
        {"class_to_idx": dict(sorted(mapping.items()))},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_class_mapping(mapping: Mapping[str, int], num_classes: int) -> dict[str, int]:
    normalized = dict(mapping)
    if len(normalized) != num_classes:
        raise ExportFormatError(
            f"class mapping size {len(normalized)} differs from model classes {num_classes}."
        )
    if any(CLASS_ID_PATTERN.fullmatch(key) is None for key in normalized):
        raise ExportFormatError("Every exported class ID must be exactly four digits.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized.values()):
        raise ExportFormatError("Every exported class index must be an integer.")
    if set(normalized.values()) != set(range(num_classes)):
        raise ExportFormatError("Exported class indices must cover [0, C) exactly.")
    return dict(sorted(normalized.items()))


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportFormatError(f"Export metadata {key} must be a positive integer.")
    return value


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportFormatError(f"Export metadata {key} must be numeric.")
    return float(value)


def _default_preprocessing_spec() -> dict[str, Any]:
    return {
        "image_size": 224,
        "resize_short_side": 256,
        "center_crop": 224,
        "input_shape": [3, 224, 224],
        "dtype": "float32",
        "normalization": "openai_clip_official",
        "test_time_augmentation": False,
    }
