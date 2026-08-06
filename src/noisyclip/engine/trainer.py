"""Protocol-driven training state machine for NoisyCLIP F02."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from noisyclip.config.schema import ProjectConfig
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.engine.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from noisyclip.engine.context import RunContext
from noisyclip.engine.evaluator import EvaluationResult, Evaluator, save_evaluation_artifacts
from noisyclip.engine.precision import NonFiniteTrainingError, PrecisionConfig, PrecisionManager
from noisyclip.losses.outputs import LossOutput
from noisyclip.models.outputs import ModelOutput
from noisyclip.models.prototypes import build_prototype_builder
from noisyclip.noise.curriculum import PartitionCurriculum
from noisyclip.noise.partition import apply_partitions, partition_by_class
from noisyclip.noise.signals import update_prediction_history
from noisyclip.noise.state import SampleState, SampleStateStore
from noisyclip.noise.trust import ClasswiseTrustAggregator
from noisyclip.submission.mapping import mapping_digest
from noisyclip.tracking.artifacts import ArtifactStore
from noisyclip.tracking.logger import JsonlLogger
from noisyclip.tracking.manifest import RunManifest
from noisyclip.utils.atomic import atomic_copy_file


class CompositeLossLike(Protocol):
    """Protocol for a configured training loss."""

    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
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
    clip_weight_metadata: Mapping[str, object] | None = None
    preprocessing_spec: Mapping[str, object] | None = None
    config_summary: Mapping[str, object] | None = None
    resume_checkpoint: Path | None = None


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
        self.early_best_metric: float | None = None
        self.epochs_without_improvement = 0

    def preflight(self) -> None:
        """Validate data, loss, trainable parameters, and run boundaries.

        Raises:
            TrainingPreflightError: If a test manifest enters training, all
            configured training losses are inactive, or trainable parameters
            violate B0/B2 rules.
        """

        _reject_test_records(self.components.train_records)
        _reject_all_losses_disabled(self.config)
        validate_trainable_parameter_set(self.components.model, _trainability_stage(self.config))
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
        if self.components.teacher is not None:
            self.components.teacher.to(self.device)
        self.components.run_manifest.transition("MODEL_READY")
        logger = JsonlLogger(self.components.artifact_store.metric("epoch_metrics.jsonl"))
        last_checkpoint = self.components.artifact_store.checkpoint("last.pt")
        exported_model: Path | None = None
        try:
            start_epoch = self._restore_if_requested()
            epochs_completed = start_epoch
            for epoch in range(start_epoch, self.config.trainer.epochs):
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
                checkpoint_improved = self._update_best(val_result)
                loss_state = _loss_state_dict(self.components.loss)
                epoch_checkpoint = save_checkpoint(
                    self.components.artifact_store.checkpoint(f"epoch_{epoch:04d}.pt"),
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
                        best_metric=self.best_metric,
                        early_best_metric=self.early_best_metric,
                        epochs_without_improvement=self.epochs_without_improvement,
                    ),
                    loss_state=loss_state,
                    minimum_free_bytes=int(self.config.tracking.minimum_free_disk_gib * 1024**3),
                )
                self.components.run_manifest.transition("CHECKPOINTED", extra={"epoch": epoch})
                self.components.state_store.commit_epoch(epoch)
                last_checkpoint = atomic_copy_file(
                    epoch_checkpoint,
                    last_checkpoint,
                    minimum_free_bytes=int(self.config.tracking.minimum_free_disk_gib * 1024**3),
                )
                if checkpoint_improved:
                    atomic_copy_file(
                        epoch_checkpoint,
                        self.components.artifact_store.checkpoint(
                            _best_checkpoint_name(self.config.evaluation.checkpoint_selection)
                        ),
                        minimum_free_bytes=int(
                            self.config.tracking.minimum_free_disk_gib * 1024**3
                        ),
                    )
                    save_evaluation_artifacts(
                        val_result, self.components.artifact_store.metric("best_eval")
                    )
                record = _epoch_record(epoch, train_stats, val_result, self.global_step)
                logger.write(record)
                epochs_completed = epoch + 1
                if (
                    self.config.trainer.early_stopping.enabled
                    and self.epochs_without_improvement
                    >= self.config.trainer.early_stopping.patience
                ):
                    break
            self._restore_best_for_export()
            exported_model = self._export_final_model()
            self.components.run_manifest.mark_done()
        except Exception as exc:
            self.components.state_store.rollback_uncommitted()
            self.components.run_manifest.mark_failed(str(exc), stage="training")
            raise TrainingFailedError(str(exc)) from exc
        return TrainResult(
            epochs_completed=epochs_completed,
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
        per_sample_logits: dict[str, Tensor] = {}
        per_sample_strong_logits: dict[str, Tensor] = {}
        per_sample_embedding: dict[str, Tensor] = {}
        loss_total = 0.0
        step_count = 0
        self.components.optimizer.zero_grad(set_to_none=True)
        _set_loader_epoch(self.components.train_loader, epoch)
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
                gradient_validator=lambda: validate_frozen_gradients(
                    self.components.model, _trainability_stage(self.config)
                ),
            )
            if stepped:
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
            logits = _get_logits(weak_output).detach().cpu().float()
            embedding = _get_embedding(weak_output).detach().cpu().float()
            strong_logits = (
                None if strong_output is None else _get_logits(strong_output).detach().cpu().float()
            )
            for index, sample_id in enumerate(batch.sample_ids):
                _store_once(per_sample_logits, sample_id, logits[index])
                _store_once(per_sample_embedding, sample_id, embedding[index])
                if strong_logits is not None:
                    _store_once(per_sample_strong_logits, sample_id, strong_logits[index])
        stepped = self.precision.step_if_needed(
            microbatch_index=step_count,
            model=self.components.model,
            optimizer=self.components.optimizer,
            force=True,
            gradient_validator=lambda: validate_frozen_gradients(
                self.components.model, _trainability_stage(self.config)
            ),
        )
        if stepped:
            if self.components.scheduler is not None:
                self.components.scheduler.step()
            self.global_step += 1
        missing = sorted(set(ordered_ids) - set(per_sample_logits))
        if missing:
            raise ValueError(f"Training epoch did not visit sample_id(s): {missing}.")
        return {
            "loss_total": loss_total / max(1, step_count),
            "per_sample_loss": {
                sample_id: per_sample_loss.get(sample_id, torch.tensor(0.0))
                for sample_id in ordered_ids
            },
            "per_sample_logits": {
                sample_id: per_sample_logits[sample_id] for sample_id in ordered_ids
            },
            "per_sample_strong_logits": {
                sample_id: per_sample_strong_logits[sample_id]
                for sample_id in ordered_ids
                if sample_id in per_sample_strong_logits
            },
            "per_sample_embedding": {
                sample_id: per_sample_embedding[sample_id] for sample_id in ordered_ids
            },
        }

    def _load_previous_states(self) -> list[SampleState]:
        sample_ids = [record.sample_id for record in self.components.train_records]
        if getattr(self.components.state_store, "latest_epoch", None) is None:
            return [_default_state(sample_id) for sample_id in sample_ids]
        return self.components.state_store.load(sample_ids)

    def _update_sample_states(
        self,
        epoch: int,
        previous_states: list[SampleState],
        train_stats: Mapping[str, object],
    ) -> list[SampleState]:
        records = self.components.train_records
        logits_by_id = _tensor_mapping(train_stats["per_sample_logits"])
        strong_logits_by_id = _tensor_mapping(train_stats["per_sample_strong_logits"])
        embeddings_by_id = _tensor_mapping(train_stats["per_sample_embedding"])
        losses_by_id = _tensor_mapping(train_stats["per_sample_loss"])
        ordered_logits = torch.stack([logits_by_id[record.sample_id] for record in records])
        history_updated = update_prediction_history(
            previous_states,
            ordered_logits,
            epoch=epoch,
            momentum=self.config.noise.signals.ema_loss.momentum,
        )
        should_update_trust = (
            self.config.noise.enabled
            and self.components.trust_aggregator is not None
            and epoch >= self.config.noise.warmup_epochs
            and (epoch - self.config.noise.warmup_epochs) % self.config.noise.update_interval_epochs
            == 0
        )
        if should_update_trust:
            trust_aggregator = self.components.trust_aggregator
            if trust_aggregator is None:  # pragma: no cover - narrowed above.
                raise RuntimeError("noise update requires a trust aggregator.")
            raw_signals = self._raw_trust_signals(
                records=records,
                previous_states=previous_states,
                logits_by_id=logits_by_id,
                strong_logits_by_id=strong_logits_by_id,
                embeddings_by_id=embeddings_by_id,
                losses_by_id=losses_by_id,
            )
            aggregated = trust_aggregator.update_epoch(
                records,
                raw_signals,
                history_updated,
                epoch,
            )
        else:
            aggregated = history_updated
        if not self.config.noise.enabled:
            return aggregated
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

    def _raw_trust_signals(
        self,
        *,
        records: list[SampleRecord],
        previous_states: list[SampleState],
        logits_by_id: Mapping[str, Tensor],
        strong_logits_by_id: Mapping[str, Tensor],
        embeddings_by_id: Mapping[str, Tensor],
        losses_by_id: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        trust_aggregator = self.components.trust_aggregator
        if trust_aggregator is None:
            raise RuntimeError("raw trust signals require a trust aggregator.")
        enabled = trust_aggregator.signal_coefficients
        raw: dict[str, Tensor] = {}
        if enabled.get("ema_loss", 0.0) > 0.0:
            momentum = self.config.noise.signals.ema_loss.momentum
            current = torch.stack([losses_by_id[record.sample_id] for record in records])
            previous = torch.tensor([state.ema_loss for state in previous_states])
            seen = torch.tensor([state.seen_count > 0 for state in previous_states])
            raw["ema_loss"] = torch.where(
                seen,
                momentum * previous + (1 - momentum) * current,
                current,
            )
        if enabled.get("prediction_stability", 0.0) > 0.0:
            values = []
            for record, state in zip(records, previous_states, strict=True):
                current = logits_by_id[record.sample_id].softmax(dim=0)
                if state.ema_probs is None:
                    values.append(torch.tensor(0.0))
                else:
                    values.append((current * torch.tensor(state.ema_probs)).sum())
            raw["prediction_stability"] = torch.stack(values)
        if enabled.get("augmentation_agreement", 0.0) > 0.0:
            missing = [r.sample_id for r in records if r.sample_id not in strong_logits_by_id]
            if missing:
                raise ValueError("augmentation_agreement requires strong views for every sample.")
            raw["augmentation_agreement"] = torch.stack(
                [
                    (
                        logits_by_id[r.sample_id].softmax(dim=0)
                        * strong_logits_by_id[r.sample_id].softmax(dim=0)
                    ).sum()
                    for r in records
                ]
            )
        prototype_names = ("prototype_similarity", "prototype_margin")
        if any(enabled.get(name, 0.0) > 0.0 for name in prototype_names):
            embeddings = torch.stack([embeddings_by_id[r.sample_id] for r in records])
            targets = torch.tensor([_target(r) for r in records], dtype=torch.int64)
            method = self.config.model.head.prototype_init.method
            if method == "multi_prototype":
                raise ValueError("multi_prototype trust signals require the U3 integration.")
            prototypes = build_prototype_builder(
                method,
                keep_fraction=self.config.model.head.prototype_init.keep_fraction,
            ).fit(embeddings, targets, None, self.components.run_context.num_classes)
            similarities = embeddings @ prototypes.T
            target_scores = similarities.gather(1, targets[:, None]).squeeze(1)
            if enabled.get("prototype_similarity", 0.0) > 0.0:
                raw["prototype_similarity"] = target_scores
            if enabled.get("prototype_margin", 0.0) > 0.0:
                masked = similarities.clone()
                masked.scatter_(1, targets[:, None], float("-inf"))
                raw["prototype_margin"] = target_scores - masked.max(dim=1).values
        return raw

    def _restore_if_requested(self) -> int:
        checkpoint = self.components.resume_checkpoint
        if checkpoint is None:
            return 0
        loss_objects: dict[str, Any] = {}
        elr = getattr(self.components.loss, "elr", None)
        if elr is not None:
            loss_objects["elr"] = elr
        metadata = load_checkpoint(
            checkpoint,
            model=self.components.model,
            optimizer=self.components.optimizer,
            scheduler=self.components.scheduler,
            precision_manager=self.precision,
            loss_objects=loss_objects,
            map_location=self.device,
        )
        if metadata.config_digest != self.components.run_context.config_digest:
            raise TrainingPreflightError("Resume checkpoint config digest mismatch.")
        if metadata.data_digest != self.components.run_context.data_digest:
            raise TrainingPreflightError("Resume checkpoint data digest mismatch.")
        latest_epoch = getattr(self.components.state_store, "latest_epoch", None)
        if metadata.sample_state_epoch != latest_epoch:
            raise TrainingPreflightError("Resume checkpoint/sample-state epoch mismatch.")
        self.global_step = metadata.global_step
        self.best_metric = metadata.best_metric
        self.early_best_metric = metadata.early_best_metric
        self.epochs_without_improvement = metadata.epochs_without_improvement
        return metadata.epoch + 1

    def _update_best(self, result: EvaluationResult) -> bool:
        value = result.metrics.get(self.config.evaluation.checkpoint_selection)
        improved = value is not None and (self.best_metric is None or value > self.best_metric)
        if improved and value is not None:
            self.best_metric = float(value)
        early_value = result.metrics.get(self.config.trainer.early_stopping.metric)
        if early_value is not None and (
            self.early_best_metric is None
            or early_value > self.early_best_metric + self.config.trainer.early_stopping.min_delta
        ):
            self.early_best_metric = float(early_value)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        save_evaluation_artifacts(result, self.components.artifact_store.metric("last_eval"))
        return improved

    def _export_final_model(self) -> Path | None:
        destination = self.components.artifact_store.artifact("model.pt")
        export = getattr(self.components.model, "export_single_model", None)
        if callable(export):
            if self.components.clip_weight_metadata is None:
                return Path(export(destination))
            return Path(
                export(
                    destination,
                    preprocessing_spec=self.components.preprocessing_spec,
                    config_summary=self.components.config_summary,
                    class_to_idx=self.components.run_context.class_to_idx,
                    mapping_digest=mapping_digest(self.components.run_context.class_to_idx),
                    clip_weight_metadata=self.components.clip_weight_metadata,
                )
            )
        return None

    def _restore_best_for_export(self) -> None:
        best_path = self.components.artifact_store.checkpoint(
            _best_checkpoint_name(self.config.evaluation.checkpoint_selection)
        )
        if not best_path.is_file():
            return
        metadata = load_checkpoint(
            best_path,
            model=self.components.model,
            map_location=self.device,
        )
        if metadata.config_digest != self.components.run_context.config_digest:
            raise TrainingPreflightError("Best checkpoint config digest mismatch before export.")
        if metadata.data_digest != self.components.run_context.data_digest:
            raise TrainingPreflightError("Best checkpoint data digest mismatch before export.")


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


def _get_embedding(output: object) -> Tensor:
    embedding = getattr(output, "embedding", None)
    if not isinstance(embedding, Tensor):
        raise ValueError("model output must expose embedding tensor.")
    if not torch.isfinite(embedding).all():
        raise NonFiniteTrainingError("model embedding contains NaN or Inf.")
    return embedding


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


def _trainability_stage(config: ProjectConfig) -> str:
    if config.model.lora.enabled:
        return "B2"
    if config.model.head.type == "cosine" or config.model.head.prototype_init.enabled:
        return "B1"
    return "B0"


def _set_loader_epoch(loader: Iterable[Batch], epoch: int) -> None:
    for component in (getattr(loader, "sampler", None), getattr(loader, "dataset", None)):
        set_epoch = getattr(component, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


def _best_checkpoint_name(metric_name: str) -> str:
    leaf = metric_name.rsplit("/", 1)[-1]
    safe = "".join(
        character if character.isalnum() or character == "_" else "_" for character in leaf
    )
    return f"best_{safe}.pt"
