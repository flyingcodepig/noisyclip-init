"""Model outputs and model-side public protocols."""

from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import (
    CosineClassifierHead,
    LinearClassifierHead,
    build_classifier_head,
)
from noisyclip.models.clip_loader import ClipWeightMetadata, LoadedClipModel, load_clip_vit_b32
from noisyclip.models.export import (
    ExportedInferenceModel,
    ExportFormatError,
    export_student_model,
    load_export_package,
    load_exported_model,
    load_exported_model_auto,
)
from noisyclip.models.lora import LoraInjectionConfig, inject_lora_into_visual_transformer
from noisyclip.models.outputs import (
    ClassifierHead,
    FrozenTeacher,
    ImageEncoder,
    ModelOutput,
    StudentModel,
)
from noisyclip.models.student import NoisyCLIPStudent
from noisyclip.models.teacher import FrozenTeacherModel

__all__ = [
    "CLIPImageBackbone",
    "ClassifierHead",
    "ClipWeightMetadata",
    "CosineClassifierHead",
    "ExportFormatError",
    "ExportedInferenceModel",
    "FrozenTeacher",
    "FrozenTeacherModel",
    "ImageEncoder",
    "LinearClassifierHead",
    "LoadedClipModel",
    "LoraInjectionConfig",
    "ModelOutput",
    "NoisyCLIPStudent",
    "StudentModel",
    "build_classifier_head",
    "export_student_model",
    "inject_lora_into_visual_transformer",
    "load_clip_vit_b32",
    "load_export_package",
    "load_exported_model",
    "load_exported_model_auto",
]
