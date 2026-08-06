"""Protocol-driven training state machine for NoisyCLIP F02."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from noisyclip.config.schema import ProjectConfig
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.engine.checkpoint import CheckpointMetadata, save_checkpoint
from noisyclip.engine.context import RunContext
from noisyclip.engine.evaluator import EvaluationResult, Evaluator, save_evaluation_artifacts
from noisyclip.engine.precision import NonFiniteTrainingError, PrecisionConfig, PrecisionManager
from noisyclip.losses.outputs import LossOutput
from noisyclip.noise.curriculum import PartitionCurriculum
from noisyclip.noise.partition import apply_partitions, partition_by_class
from noisyclip.noise.state import SampleState, SampleStateStore
from noisyclip.noise.trust import ClasswiseTrustAggregator
from noisyclip.tracking.artifacts import ArtifactStore
from noisyclip.tracking.logger import JsonlLogger
from noisyclip.tracking.manifest import RunManifest


class CompositeLossLike(Protocol):
    """Protocol for a configured training loss."""

    def __call__(
        self,
        batch: Batch,
        student_weak: object,
        student_strong: object | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> LossOutput:
        """Return a finite scalar total plus optional per-sample loss."""


@dataclass(slots=True)
class TrainerComponents:
    """Dependency-injected components used by the training engine."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss: CompositeLossLike
    train_loader: Iterable[Batch]
    val_loader: Iterable[Batch]
    train_records: list[SampleRecord]
    state_store: SampleStateStore
    run_context: RunContext
    artifact_store: ArtifactStore
    run_manifest: RunManifest
    scheduler: Any | None = None
    teacher: Any | None = None
    trust_aggregator: ClasswiseTrustAggregator | None = None
    curriculum: PartitionCurriculum | None = None


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary of a completed training run."""

    epochs_completed: int
    global_step: int
    best_metric: float | None
    last_checkpoint: Path
    exported_model: Path | None


class TrainingPreflightError(ValueError):
    """Raised before training starts for config/data/compliance errors."""


class TrainingFailedError(RuntimeError):
    """Raised when the state machine enters FAILED during training."""


class Trainer:
    """Run the F02 training state machine over injected components.

    Args:
        config: Immutable project config.
        components: Model, optimizer, loaders, state store, and tracking
            objects. Loaders yield public `Batch` records.
        device: Torch device. CPU/fp32 is supported for tests.

    Raises:
        TrainingPreflightError: If data, loss, or trainable-parameter guards
            fail before the first backward pass.
    """

    def __init__(
        self,
        *,
        config: ProjectConfig,
        components: TrainerComponents,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.components = components
        self.device = torch.device(device)
        self.precision = PrecisionManager(
            PrecisionConfig(
                precision=config.trainer.precision,
                gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
                gradient_clip_norm=config.trainer.gradient_clip_norm,
            ),
            device=self.device,
        )
        self.global_step = 0
        self.best_metric: float | None = None

    def preflight(self) -> None:
        """Validate data, loss, trainable parameters, and run boundaries.

        Raises:
            TrainingPreflightError: If a test manifest enters training, all
            configured training losses are inactive, or trainable parameters
            violate B0/B2 rules.
        """

        _reject_test_records(self.components.train_records)
        _reject_all_losses_disabled(self.config)
        validate_trainable_parameter_set(self.components.model, self.config.experiment.stage)
        self.components.run_manifest.transition("PREFLIGHT_OK")

    def fit(self) -> TrainResult:
        """Execute training, validation, checkpoint, state commit, and export.

        Returns:
            `TrainResult` with the final checkpoint and exported model path.

        Raises:
            TrainingFailedError: If a training, checkpoint, state, or export
            error occurs. A `FAILED` marker is written before re-raising.
        """

        self.preflight()
        self.components.run_manifest.transition("DATA_READY")
        self.components.model.to(self.device)
        self.components.run_manifest.transition("MODEL_READY")
        logger = JsonlLogger(self.components.artifact_store.metric("epoch_metrics.jsonl"))
        last_checkpoint = self.components.artifact_store.checkpoint("last.pt")
        exported_model: Path | None = None
        try:
            for epoch in range(self.config.trainer.epochs):
                self.components.run_manifest.transition("TRAINING", extra={"epoch": epoch})
                previous_states = self._load_previous_states()
                train_stats = self._train_epoch(epoch, previous_states)
                self.components.run_manifest.transition("VALIDATING", extra={"epoch": epoch})
                val_result = Evaluator(
                    model=self.components.model,
                    num_classes=self.components.run_context.num_classes,
                    device=self.device,
                ).evaluate(self.components.val_loader)
                new_states = self._update_sample_states(epoch, previous_states, train_stats)
                self.components.state_store.stage_epoch(new_states, epoch)
                loss_state = _loss_state_dict(self.components.loss)
                last_checkpoint = save_checkpoint(
                    last_checkpoint,
                    model=self.components.model,
                    optimizer=self.components.optimizer,
                    scheduler=self.components.scheduler,
                    scaler_state=self.precision.state_dict(),
                    metadata=CheckpointMetadata(
                        epoch=epoch,
                        global_step=self.global_step,
                        sample_state_epoch=epoch,
                        config_digest=self.components.run_context.config_digest,
                        data_digest=self.components.run_context.data_digest,
                    ),
                    loss_state=loss_state,
                    minimum_free_bytes=int(self.config.tracking.minimum_free_disk_gib * 1024**3),
                )
                self.components.run_manifest.transition("CHECKPOINTED", extra={"epoch": epoch})
                self.components.state_store.commit_epoch(epoch)
                record = _epoch_record(epoch, train_stats, val_result, self.global_step)
                logger.write(record)
                self._update_best(val_result)
            exported_model = self._export_final_model()
            self.components.run_manifest.mark_done()
        except Exception as exc:
            self.components.state_store.rollback_uncommitted()
            self.components.run_manifest.mark_failed(str(exc), stage="training")
            raise TrainingFailedError(str(exc)) from exc
        return TrainResult(
            epochs_completed=self.config.trainer.epochs,
            global_step=self.global_step,
            best_metric=self.best_metric,
            last_checkpoint=last_checkpoint,
            exported_model=exported_model,
        )

    def _train_epoch(
        self,
        epoch: int,
        previous_states: list[SampleState],
    ) -> dict[str, Tensor | float | dict[str, Tensor]]:
        self.components.model.train()
        by_id = {state.sample_id: state for state in previous_states}
        ordered_ids = [record.sample_id for record in self.components.train_records]
        per_sample_loss: dict[str, Tensor] = {}
        per_sample_probs: dict[str, Tensor] = {}
        loss_total = 0.0
        step_count = 0
        self.components.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(self.components.train_loader):
            _reject_test_batch(batch)
            states = [by_id[sample_id] for sample_id in batch.sample_ids]
            batch = _batch_to_device(batch, self.device)
            with self.precision.autocast():
                weak_output = self.components.model(batch.image_weak)
                strong_output = (
                    self.components.model(batch.image_strong)
                    if batch.image_strong is not None
                    else None
                )
                teacher_embedding = _teacher_embedding(self.components.teacher, batch)
                loss = self.components.loss(
                    batch,
                    weak_output,
                    strong_output,
                    teacher_embedding,
                    states,
                    epoch,
                )
            self.precision.backward(loss.total)
            stepped = self.precision.step_if_needed(
                microbatch_index=batch_index,
                model=self.components.model,
                optimizer=self.components.optimizer,
            )
            if stepped:
                validate_frozen_gradients(self.components.model, self.config.experiment.stage)
                if self.components.scheduler is not None:
                    self.components.scheduler.step()
                self.global_step += 1
            loss_total += float(loss.total.detach().cpu().item())
            step_count += 1
            if loss.per_sample_supervised is not None:
                detached_loss = loss.per_sample_supervised.detach().cpu().float()
                if detached_loss.shape != (len(batch.sample_ids),):
                    raise ValueError("per_sample_supervised must have shape [B].")
                for index, sample_id in enumerate(batch.sample_ids):
                    _store_once(per_sample_loss, sample_id, detached_loss[index])
            probabilities = torch.softmax(_get_logits(weak_output).detach().cpu().float(), dim=1)
            for index, sample_id in enumerate(batch.sample_ids):
                _store_once(per_sample_probs, sample_id, probabilities[index])
        missing = sorted(set(ordered_ids) - set(per_sample_probs))
        if missing:
            raise ValueError(f"Training epoch did not visit sample_id(s): {missing}.")
        return {
            "loss_total": loss_total / max(1, step_count),
            "per_sample_loss": {
                sample_id: per_sample_loss.get(sample_id, torch.tensor(0.0))
                for sample_id in ordered_ids
            },
            "per_sample_probs": {
                sample_id: per_sample_probs[sample_id] for sample_id in ordered_ids
            },
        }

    def _load_previous_states(self) -> list[SampleState]:
        sample_ids = [record.sample_id for record in self.components.train_records]
        try:
            return self.components.state_store.load(sample_ids)
        except ValueError:
            return [_default_state(sample_id) for sample_id in sample_ids]

    def _update_sample_states(
        self,
        epoch: int,
        previous_states: list[SampleState],
        train_stats: Mapping[str, object],
    ) -> list[SampleState]:
        records = self.components.train_records
        previous_by_id = {state.sample_id: state for state in previous_states}
        probs_by_id = _tensor_mapping(train_stats["per_sample_probs"])
        losses_by_id = _tensor_mapping(train_stats["per_sample_loss"])
        history_updated = [
            _update_prediction_history(
                previous_by_id[record.sample_id],
                probs_by_id[record.sample_id],
                losses_by_id[record.sample_id],
                epoch,
            )
            for record in records
        ]
        if self.components.trust_aggregator is not None:
            raw_signals = {
                "ema_loss": torch.stack([losses_by_id[record.sample_id] for record in records]),
                "prediction_stability": torch.tensor(
                    [state.prediction_stability for state in history_updated],
                    dtype=torch.float32,
                ),
            }
            aggregated = self.components.trust_aggregator.update_epoch(
                records,
                raw_signals,
                history_updated,
                epoch,
            )
        else:
            aggregated = history_updated
        targets = torch.tensor([_target(record) for record in records], dtype=torch.int64)
        trust_scores = torch.tensor(
            [state.trust_score for state in aggregated],
            dtype=torch.float32,
        )
        partitions = partition_by_class(
            [record.sample_id for record in records],
            targets,
            trust_scores,
            trusted_quantile=self.config.noise.partition.trusted_quantile or 0.65,
            uncertain_quantile=self.config.noise.partition.uncertain_quantile or 0.90,
            min_samples_per_class=self.config.noise.partition.min_samples_per_class,
        )
        partitioned = apply_partitions(aggregated, partitions, epoch=epoch)
        if self.components.curriculum is not None:
            return self.components.curriculum.apply(partitioned, epoch)
        return partitioned

    def _update_best(self, result: EvaluationResult) -> None:
        value = result.metrics.get(self.config.evaluation.checkpoint_selection)
        if value is not None and (self.best_metric is None or value > self.best_metric):
            self.best_metric = float(value)
        save_evaluation_artifacts(result, self.components.artifact_store.metric("last_eval"))

    def _export_final_model(self) -> Path | None:
        destination = self.components.artifact_store.artifact("model.pt")
        export = getattr(self.components.model, "export_single_model", None)
        if callable(export):
            return Path(export(destination))
        return None


def validate_trainable_parameter_set(model: nn.Module, stage: str) -> None:
    """Validate actual trainable parameter names for B0/B1/B2.

    Args:
        model: Student model with named parameters.
        stage: Experiment stage string. `init`, `B0`, and `B1` freeze backbone;
            `B2` allows head and LoRA only.

    Raises:
        TrainingPreflightError: If unauthorized parameters are trainable or no
            parameters are trainable.
    """

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise TrainingPreflightError("No trainable parameters were found.")
    normalized_stage = "B0" if stage == "init" else stage
    if normalized_stage in {"B0", "B1"}:
        forbidden = [name for name in trainable if name.startswith("backbone.")]
    elif normalized_stage == "B2":
        forbidden = [
            name for name in trainable if not (name.startswith("head.") or ".lora_" in f".{name}")
        ]
    else:
        forbidden = []
    if forbidden:
        raise TrainingPreflightError(f"Unauthorized trainable parameters: {forbidden}.")


def validate_frozen_gradients(model: nn.Module, stage: str) -> None:
    """Require frozen backbone gradients to be absent or exactly zero.

    Args:
        model: Student model after backward.
        stage: Stage name controlling freeze policy.

    Raises:
        NonFiniteTrainingError: If a forbidden gradient is non-zero.
    """

    normalized_stage = "B0" if stage == "init" else stage
    if normalized_stage not in {"B0", "B1"}:
        return
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.") and parameter.grad is not None:
            if not torch.equal(parameter.grad, torch.zeros_like(parameter.grad)):
                raise NonFiniteTrainingError(f"B0/B1 backbone gradient is non-zero: {name}.")


def _reject_test_records(records: Sequence[SampleRecord]) -> None:
    if not records:
        raise TrainingPreflightError("train_records must be non-empty.")
    for record in records:
        if record.split == "test" or record.target is None:
            raise TrainingPreflightError(
                f"Test/unlabeled sample cannot enter training: {record.sample_id}."
            )


def _reject_test_batch(batch: Batch) -> None:
    if batch.targets is None or batch.class_ids is None:
        raise TrainingPreflightError("Training batch cannot be unlabeled or test-derived.")


def _reject_all_losses_disabled(config: ProjectConfig) -> None:
    active = (
        config.loss.cross_entropy.enabled and config.loss.cross_entropy.weight > 0.0,
        config.loss.elr.enabled and config.loss.elr.weight > 0.0,
        config.loss.consistency.enabled and config.loss.consistency.weight > 0.0,
        config.loss.feature_anchor.enabled and config.loss.feature_anchor.weight > 0.0,
    )
    if not any(active):
        raise TrainingPreflightError(
            "At least one training loss must be enabled with positive weight."
        )


def _batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        sample_ids=batch.sample_ids,
        image_weak=batch.image_weak.to(device),
        image_strong=None if batch.image_strong is None else batch.image_strong.to(device),
        targets=None if batch.targets is None else batch.targets.to(device),
        class_ids=batch.class_ids,
    )


def _teacher_embedding(teacher: Any | None, batch: Batch) -> Tensor | None:
    if teacher is None:
        return None
    with torch.no_grad():
        return teacher.encode_image(batch.image_weak)


def _get_logits(output: object) -> Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise ValueError("model output must expose logits tensor.")
    if not torch.isfinite(logits).all():
        raise NonFiniteTrainingError("model logits contain NaN or Inf.")
    return logits


def _store_once(store: dict[str, Tensor], sample_id: str, value: Tensor) -> None:
    if sample_id in store:
        raise ValueError(f"sample_id was seen more than once in the same epoch: {sample_id}.")
    store[sample_id] = value.detach().cpu().clone()


def _default_state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=0,
        ema_loss=0.0,
        ema_probs=None,
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=1.0,
        prototype_margin=1.0,
        trust_score=1.0,
        supervised_weight=1.0,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )


def _update_prediction_history(
    state: SampleState,
    probabilities: Tensor,
    supervised_loss: Tensor,
    epoch: int,
) -> SampleState:
    if probabilities.ndim != 1 or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite [C].")
    if bool((probabilities < 0).any()) or abs(float(probabilities.sum().item()) - 1.0) > 1e-4:
        raise ValueError("probabilities must be in [0, 1] and sum to 1.")
    previous = state.ema_probs
    stability = 1.0
    if previous is not None:
        previous_tensor = torch.tensor(previous, dtype=torch.float32)
        if previous_tensor.shape != probabilities.shape:
            raise ValueError("previous ema_probs class count differs from current probabilities.")
        stability_tensor = (1.0 - 0.5 * torch.abs(probabilities - previous_tensor).sum()).clamp(
            0,
            1,
        )
        stability = float(stability_tensor)
    return replace(
        state,
        seen_count=state.seen_count + 1,
        ema_loss=float(supervised_loss.detach().cpu().item()),
        ema_probs=[float(item) for item in probabilities.tolist()],
        prediction_stability=stability,
        updated_epoch=epoch,
    )


def _target(record: SampleRecord) -> int:
    if record.target is None:
        raise TrainingPreflightError(f"record target is missing: {record.sample_id}.")
    return record.target


def _tensor_mapping(raw: object) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping):
        raise TypeError("train stat mapping must be a mapping.")
    result: dict[str, Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError("train stat mappings must be str -> Tensor.")
        result[key] = value
    return result


def _loss_state_dict(loss: object) -> dict[str, object]:
    state: dict[str, object] = {}
    elr = getattr(loss, "elr", None)
    state_dict = getattr(elr, "state_dict", None)
    if callable(state_dict):
        state["elr"] = state_dict()
    return state


def _epoch_record(
    epoch: int,
    train_stats: Mapping[str, object],
    val_result: EvaluationResult,
    global_step: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        "train/loss_total": _float_stat(train_stats["loss_total"]),
    }
    record.update(val_result.metrics)
    for name, reason in val_result.metric_reasons.items():
        record[f"{name}/reason"] = reason
    return record


def _float_stat(value: object) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"stat value must be numeric, got {type(value).__name__}.")
