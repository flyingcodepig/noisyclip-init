"""Strict Pydantic schema for the F01 configuration contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown fields and freezes validated configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictConfigModel):
    """Experiment section placeholder for future tracked experiment metadata."""


class PathsConfig(StrictConfigModel):
    """Paths section placeholder; later modules must add explicit path fields."""


class DataConfig(StrictConfigModel):
    """Data section placeholder; later modules must add explicit data fields."""


class ModelConfig(StrictConfigModel):
    """Model section placeholder; later modules must add explicit model fields."""


class NoiseConfig(StrictConfigModel):
    """Noise section placeholder; later modules must add explicit noise fields."""


class LossConfig(StrictConfigModel):
    """Loss section placeholder; later modules must add explicit loss fields."""


class TrainerConfig(StrictConfigModel):
    """Trainer section placeholder; later modules must add explicit trainer fields."""


class EvaluationConfig(StrictConfigModel):
    """Evaluation section placeholder; later modules must add explicit metric fields."""


class TrackingConfig(StrictConfigModel):
    """Tracking section placeholder; later modules must add explicit artifact fields."""


class SubmissionConfig(StrictConfigModel):
    """Submission section placeholder; later modules must add explicit CSV fields."""


class ProjectConfig(StrictConfigModel):
    """Top-level configuration with the ten fixed architecture sections."""

    experiment: ExperimentConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    noise: NoiseConfig
    loss: LossConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    submission: SubmissionConfig
