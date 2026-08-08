"""Strict, immutable configuration schema for all declared experiment modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown fields and freezes validated configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictConfigModel):
    """Identity and research metadata for one reproducible experiment."""

    name: str = "base"
    stage: Literal["init"] = "init"
    seed: int = Field(default=20260806, ge=0)
    baseline_run_id: str | None = None
    research_question: str = "Base configuration; do not run directly."
    tags: tuple[str, ...] = ()


class PathsConfig(StrictConfigModel):
    """Raw-data, run, cache, and generated-manifest locations."""

    train_root: str = "${oc.env:NOISYCLIP_TRAIN_ROOT}"
    test_root: str = "${oc.env:NOISYCLIP_TEST_ROOT}"
    run_root: str = "${oc.env:NOISYCLIP_RUN_ROOT}"
    cache_root: str = "${oc.env:XDG_CACHE_HOME}"
    train_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    class_mapping: str | None = None


class TrainTransformConfig(StrictConfigModel):
    """Conservative fine-grained training augmentation."""

    random_resized_crop_scale: tuple[float, float] = (0.75, 1.0)
    horizontal_flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    color_jitter_strength: float = Field(default=0.1, ge=0.0)


class StrongTransformConfig(StrictConfigModel):
    """Optional stronger view used by consistency training."""

    enabled: bool = False
    random_resized_crop_scale: tuple[float, float] = (0.70, 1.0)
    randaugment_magnitude: int = Field(default=5, ge=0)


class EvalTransformConfig(StrictConfigModel):
    """Deterministic validation and test preprocessing."""

    resize_short_side: int = Field(default=256, ge=224)
    center_crop: int = Field(default=224, ge=1)


class DataConfig(StrictConfigModel):
    """Dataset layout, audit, split, and preprocessing configuration."""

    expected_num_classes: int = Field(default=500, ge=1)
    class_id_regex: Literal["^[0-9]{4}$"] = "^[0-9]{4}$"
    val_fraction: float = Field(default=0.10, gt=0.0, lt=0.5)
    split_seed: int = Field(default=20260806, ge=0)
    image_size: Literal[224] = 224
    allow_truncated_images: bool = True
    unreadable_policy: Literal["fail_audit", "skip_with_record"] = "fail_audit"
    hash_files: bool = True
    decode_backend: Literal["pillow", "torchvision_fallback"] = "torchvision_fallback"
    normalize_on_device: bool = True
    train_transform: TrainTransformConfig = Field(default_factory=TrainTransformConfig)
    strong_transform: StrongTransformConfig = Field(default_factory=StrongTransformConfig)
    eval_transform: EvalTransformConfig = Field(default_factory=EvalTransformConfig)


class BackboneConfig(StrictConfigModel):
    """Allowlisted official CLIP backbone selection."""

    name: Literal["ViT-B/32"] = "ViT-B/32"
    pretrained: Literal["openai"] = "openai"
    source_allowlist: tuple[Literal["openai_clip_official"], ...] = ("openai_clip_official",)
    freeze: bool = True


class PrototypeInitConfig(StrictConfigModel):
    """Classifier prototype initialization and optional multi-prototype settings."""

    enabled: bool = False
    method: Literal["trimmed_mean", "multi_prototype"] = "trimmed_mean"
    keep_fraction: float = Field(default=0.80, gt=0.0, le=1.0)
    prototypes_per_class: int | None = Field(default=None, ge=1)
    minimum_samples_per_prototype: int | None = Field(default=None, ge=1)


class HeadConfig(StrictConfigModel):
    """Linear or cosine classifier head configuration."""

    type: Literal["linear", "cosine"] = "linear"
    temperature_init: float | None = Field(default=None, gt=0.0)
    temperature_min: float | None = Field(default=None, gt=0.0)
    temperature_max: float | None = Field(default=None, gt=0.0)
    prototype_init: PrototypeInitConfig = Field(default_factory=PrototypeInitConfig)


class LoraConfig(StrictConfigModel):
    """Visual-transformer LoRA injection policy."""

    enabled: bool = False
    target_blocks: tuple[int, ...] = ()
    target_projections: tuple[Literal["q", "k", "v"], ...] = ()
    rank: int = Field(default=0, ge=0)
    alpha: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class TeacherConfig(StrictConfigModel):
    """Training-only frozen teacher policy."""

    enabled: bool = False
    inference_export: Literal[False] = False


class ModelConfig(StrictConfigModel):
    """Backbone, classifier, LoRA, and teacher configuration."""

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)


class EmaLossSignalConfig(StrictConfigModel):
    """EMA loss trust signal."""

    enabled: bool = False
    coefficient: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)


class SimpleSignalConfig(StrictConfigModel):
    """Trust signal with only an enable flag and coefficient."""

    enabled: bool = False
    coefficient: float = Field(default=0.0, ge=0.0)


class StabilitySignalConfig(SimpleSignalConfig):
    """Prediction stability signal over a fixed epoch window."""

    window: int = Field(default=3, ge=1)


class TrustSignalsConfig(StrictConfigModel):
    """All raw trust signals and their aggregation coefficients."""

    ema_loss: EmaLossSignalConfig = Field(default_factory=EmaLossSignalConfig)
    augmentation_agreement: SimpleSignalConfig = Field(default_factory=SimpleSignalConfig)
    prototype_similarity: SimpleSignalConfig = Field(default_factory=SimpleSignalConfig)
    prototype_margin: SimpleSignalConfig = Field(default_factory=SimpleSignalConfig)
    prediction_stability: StabilitySignalConfig = Field(default_factory=StabilitySignalConfig)


class PartitionConfig(StrictConfigModel):
    """Class-aware fixed or adaptive sample partitioning."""

    method: Literal["fixed_quantile", "adaptive_mixture"] = "fixed_quantile"
    trusted_quantile: float | None = Field(default=0.65, ge=0.0, le=1.0)
    uncertain_quantile: float | None = Field(default=0.90, ge=0.0, le=1.0)
    min_samples_per_class: int = Field(default=2, ge=1)
    threshold_ema_momentum: float | None = Field(default=None, ge=0.0, lt=1.0)
    maximum_epoch_change: float | None = Field(default=None, gt=0.0, le=1.0)


class TrustWeightsConfig(StrictConfigModel):
    """Continuous supervised weights assigned to trust partitions."""

    trusted: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertain_min: float = Field(default=0.3, ge=0.0, le=1.0)
    uncertain_max: float = Field(default=0.7, ge=0.0, le=1.0)
    suspicious: float = Field(default=0.1, ge=0.0, le=1.0)


class CurriculumConfig(StrictConfigModel):
    """Optional trusted-to-uncertain sample admission schedule."""

    enabled: bool = False
    trusted_start_epoch: int | None = Field(default=None, ge=0)
    uncertain_start_epoch: int | None = Field(default=None, ge=0)
    suspicious_start_epoch: int | None = Field(default=None, ge=0)
    ramp_epochs: int | None = Field(default=None, ge=1)


class PseudolabelConfig(StrictConfigModel):
    """Strictly gated soft pseudo-label policy."""

    enabled: bool = False
    start_epoch: int = Field(default=999, ge=0)
    confidence_threshold: float = Field(default=0.98, gt=0.0, le=1.0)
    stability_window: int = Field(default=5, ge=1)
    minimum_prototype_margin: float = 0.20
    maximum_dataset_fraction: float = Field(default=0.05, gt=0.0, le=0.25)
    pseudo_mix: float = Field(default=0.8, gt=0.0, lt=1.0)


class GradientProjectionConfig(StrictConfigModel):
    """Optional trust-aligned gradient projection experiment."""

    enabled: bool = False
    parameter_scope: Literal["lora_and_head"] = "lora_and_head"
    trusted_reference_quantile: float = Field(default=0.65, ge=0.0, le=1.0)
    maximum_projection_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class NoiseConfig(StrictConfigModel):
    """Noise modeling, state update, partition, curriculum, and pseudo-label settings."""

    enabled: bool = False
    warmup_epochs: int = Field(default=3, ge=0)
    update_interval_epochs: int = Field(default=3, ge=1)
    normalize_within_class: bool = True
    signals: TrustSignalsConfig = Field(default_factory=TrustSignalsConfig)
    partition: PartitionConfig = Field(default_factory=PartitionConfig)
    weights: TrustWeightsConfig = Field(default_factory=TrustWeightsConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    pseudolabel: PseudolabelConfig = Field(default_factory=PseudolabelConfig)
    gradient_projection: GradientProjectionConfig = Field(default_factory=GradientProjectionConfig)


class CrossEntropyConfig(StrictConfigModel):
    """Weighted supervised cross-entropy settings."""

    enabled: bool = True
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    weight: float = Field(default=1.0, ge=0.0)


class ElrConfig(StrictConfigModel):
    """Early-learning regularization settings."""

    enabled: bool = False
    weight: float = Field(default=0.0, ge=0.0)
    start_epoch: int = Field(default=999, ge=0)
    target_momentum: float = Field(default=0.7, ge=0.0, lt=1.0)


class ConsistencyConfig(StrictConfigModel):
    """Weak/strong prediction consistency settings."""

    enabled: bool = False
    weight: float = Field(default=0.0, ge=0.0)
    start_epoch: int = Field(default=999, ge=0)
    temperature: float = Field(default=1.0, gt=0.0)


class FeatureAnchorConfig(StrictConfigModel):
    """Frozen-teacher feature anchoring settings."""

    enabled: bool = False
    weight: float = Field(default=0.0, ge=0.0)
    metric: Literal["cosine"] = "cosine"


class LogitAdjustmentConfig(StrictConfigModel):
    """Conditional class-prior logit adjustment settings."""

    enabled: bool = False
    tau: float = Field(default=0.0, ge=0.0)
    count_source: Literal["trusted_effective_count"] | None = None


class LossConfig(StrictConfigModel):
    """Composite loss configuration."""

    cross_entropy: CrossEntropyConfig = Field(default_factory=CrossEntropyConfig)
    elr: ElrConfig = Field(default_factory=ElrConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    feature_anchor: FeatureAnchorConfig = Field(default_factory=FeatureAnchorConfig)
    logit_adjustment: LogitAdjustmentConfig = Field(default_factory=LogitAdjustmentConfig)


class OptimizerConfig(StrictConfigModel):
    """Optimizer and per-parameter-group learning rates."""

    name: Literal["adamw"] = "adamw"
    head_lr: float = Field(default=0.001, gt=0.0)
    lora_lr: float = Field(default=0.00005, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)


class SchedulerConfig(StrictConfigModel):
    """Learning-rate scheduler settings."""

    name: Literal["cosine"] = "cosine"
    warmup_epochs: int = Field(default=2, ge=0)
    min_lr_ratio: float = Field(default=0.01, ge=0.0, le=1.0)


class EarlyStoppingConfig(StrictConfigModel):
    """Validation-based early stopping settings."""

    enabled: bool = True
    metric: str = "val/top1"
    patience: int = Field(default=8, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)


class FrozenFeatureCacheConfig(StrictConfigModel):
    """Reusable exact frozen-CLIP features for B0/B1 only."""

    enabled: bool = False
    directory: str | None = None
    verify_hashes: bool = True


class TrainerConfig(StrictConfigModel):
    """Training budget, precision, loading, optimization, and checkpoint settings."""

    epochs: int = Field(default=30, ge=1)
    device: str = "cuda:0"
    precision: Literal["amp_fp16", "fp32"] = "amp_fp16"
    batch_size: int = Field(default=128, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_workers: int = Field(default=8, ge=0)
    pin_memory: bool = True
    prefetch_factor: int = Field(default=4, ge=1)
    persistent_workers: bool = True
    runtime_tensor_checks: Literal["full", "boundary"] = "boundary"
    frozen_feature_cache: FrozenFeatureCacheConfig = Field(default_factory=FrozenFeatureCacheConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    deterministic: bool = True
    checkpoint_every_epochs: int = Field(default=1, ge=1)
    keep_last_checkpoints: int = Field(default=2, ge=1)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class EvaluationConfig(StrictConfigModel):
    """Validation metrics and checkpoint-selection policy."""

    metrics: tuple[str, ...] = (
        "top1",
        "macro_accuracy",
        "per_class_accuracy",
        "bottom_quartile_accuracy",
        "trusted_top1",
        "augmentation_agreement",
        "feature_cosine_to_base",
    )
    checkpoint_selection: str = "val/top1"
    test_time_augmentation: Literal[False] = False


class TrackingConfig(StrictConfigModel):
    """Machine-readable logs and artifact retention policy."""

    jsonl: bool = True
    tensorboard: bool = True
    save_sample_state_every_update: bool = True
    save_confusion_matrix: bool = True
    fail_if_run_exists: bool = True
    minimum_free_disk_gib: float = Field(default=20.0, gt=0.0)


class SubmissionConfig(StrictConfigModel):
    """Competition CSV format constraints."""

    filename: Literal["pred_results.csv"] = "pred_results.csv"
    include_header: Literal[False] = False
    expected_columns: tuple[Literal["filename", "class_id"], ...] = (
        "filename",
        "class_id",
    )
    class_id_regex: Literal["^[0-9]{4}$"] = "^[0-9]{4}$"


class ProjectConfig(StrictConfigModel):
    """Resolved top-level configuration consumed by every CLI entry point."""

    schema_version: Literal[1] = 1
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)
