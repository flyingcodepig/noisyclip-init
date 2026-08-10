"""Protocol-driven training state machine for NoisyCLIP F02."""

from __future__ import annotations

import json
import math
import shutil
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn

from noisyclip.config.schema import ProjectConfig
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.engine.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from noisyclip.engine.context import RunContext
from noisyclip.engine.device import BatchDeviceIterator
from noisyclip.engine.evaluator import EvaluationResult, Evaluator, save_evaluation_artifacts
from noisyclip.engine.precision import NonFiniteTrainingError, PrecisionConfig, PrecisionManager
from noisyclip.losses.outputs import LossOutput
from noisyclip.models.export import load_export_package
from noisyclip.models.outputs import ModelOutput
from noisyclip.models.prototypes import build_prototype_builder
from noisyclip.noise.curriculum import PartitionCurriculum
from noisyclip.noise.partition import (
    apply_partitions,
    apply_supervision_weights,
    partition_by_class,
)
from noisyclip.noise.signals import (
    prediction_stability_from_history,
    update_prediction_history_from_top1,
)
from noisyclip.noise.state import SampleState, SampleStateStore
from noisyclip.noise.trust import ClasswiseTrustAggregator
from noisyclip.submission.mapping import mapping_digest
from noisyclip.tracking.artifacts import ArtifactStore
from noisyclip.tracking.logger import JsonlLogger
from noisyclip.tracking.manifest import RunManifest
from noisyclip.utils.atomic import atomic_copy_file, atomic_save_with_writer, atomic_write_bytes
from noisyclip.utils.runtime_checks import tensor_value_checks


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
    base_val_embeddings: Mapping[str, Tensor] | None = None
    reference_cache_signature: str | None = None


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
        self.previous_feature_cosine: float | None = None

    def preflight(self) -> None:
        """Validate data, loss, trainable parameters, and run boundaries.

        Raises:
            TrainingPreflightError: If a test manifest enters training, all
            configured training losses are inactive, or trainable parameters
            violate B0/B2 rules.
        """

        _reject_test_records(self.components.train_records)
        _reject_all_losses_disabled(self.config)
        stage = _trainability_stage(self.config)
        validate_trainable_parameter_set(self.components.model, stage)
        if stage == "B2":
            if self.components.base_val_embeddings is None:
                raise TrainingPreflightError(
                    "B2 requires provenance-bound validation reference embeddings."
                )
            if not self.config.evaluation.feature_drift_guard.enabled:
                raise TrainingPreflightError("B2 requires the feature drift guard.")
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
            if start_epoch > 0:
                self._restore_feature_cosine_history()
            epochs_completed = start_epoch
            for epoch in range(start_epoch, self.config.trainer.epochs):
                self.components.run_manifest.transition("TRAINING", extra={"epoch": epoch})
                previous_states = self._load_previous_states()
                train_stats = self._train_epoch(epoch, previous_states)
                self.components.run_manifest.transition("VALIDATING", extra={"epoch": epoch})
                validation_started = time.perf_counter()
                val_result = Evaluator(
                    model=self.components.model,
                    num_classes=self.components.run_context.num_classes,
                    device=self.device,
                    runtime_tensor_checks=self.config.trainer.runtime_tensor_checks,
                ).evaluate(
                    self.components.val_loader,
                    base_embeddings=self.components.base_val_embeddings,
                )
                _synchronize_if_cuda(self.device)
                train_stats["validation_seconds"] = time.perf_counter() - validation_started
                self._guard_feature_drift(val_result)
                sample_state_epoch: int | None = None
                if self.config.noise.enabled:
                    new_states = self._update_sample_states(epoch, previous_states, train_stats)
                    self.components.state_store.stage_epoch(new_states, epoch)
                    sample_state_epoch = epoch
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
                        sample_state_epoch=sample_state_epoch,
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
                if sample_state_epoch is not None:
                    self.components.state_store.commit_epoch(sample_state_epoch)
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
                if checkpoint_improved:
                    _write_json(
                        self.components.artifact_store.metric("best_metrics.json"),
                        record,
                        overwrite=True,
                    )
                epochs_completed = epoch + 1
                if (
                    self.config.trainer.early_stopping.enabled
                    and self.epochs_without_improvement
                    >= self.config.trainer.early_stopping.patience
                ):
                    break
            self._restore_best_for_export()
            self._save_final_prototypes()
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
    ) -> dict[str, object]:
        validate_trainable_parameter_set(self.components.model, _trainability_stage(self.config))
        self.components.model.train()
        by_id = {state.sample_id: state for state in previous_states}
        track_sample_state = self.config.noise.enabled
        ordered_ids = (
            [record.sample_id for record in self.components.train_records]
            if track_sample_state
            else []
        )
        collect_trust_signals = track_sample_state and self._trust_update_due(epoch)
        collect_loss = collect_trust_signals and self.config.noise.signals.ema_loss.enabled
        collect_embeddings = collect_trust_signals and (
            self.config.noise.signals.prototype_similarity.enabled
            or self.config.noise.signals.prototype_margin.enabled
        )
        collect_agreement = (
            collect_trust_signals
            and self.config.noise.signals.augmentation_agreement.enabled
        )
        per_sample_loss: dict[str, Tensor] = {}
        per_sample_logits: dict[str, Tensor] = {}
        per_sample_strong_logits: dict[str, Tensor] = {}
        per_sample_embedding: dict[str, Tensor] = {}
        epoch_sample_ids: list[str] = []
        prediction_parts: list[Tensor] = []
        loss_total = torch.zeros((), device=self.device)
        component_totals: dict[str, Tensor] = {}
        correct_total = torch.zeros((), device=self.device, dtype=torch.int64)
        observed_total = 0
        gradient_norms: list[Tensor] = []
        step_count = 0
        epoch_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.components.optimizer.zero_grad(set_to_none=True)
        _set_loader_epoch(self.components.train_loader, epoch)
        device_batches = BatchDeviceIterator(self.components.train_loader, self.device)
        for batch_index, batch in enumerate(device_batches):
            _reject_test_batch(batch)
            states = [by_id[sample_id] for sample_id in batch.sample_ids]
            checks_enabled = self.config.trainer.runtime_tensor_checks == "full" or batch_index == 0
            with tensor_value_checks(enabled=checks_enabled), self.precision.autocast():
                weak_output = _forward_batch(self.components.model, batch, strong=False)
                strong_output = (
                    _forward_batch(self.components.model, batch, strong=True)
                    if batch.image_strong is not None or batch.embedding_strong is not None
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
                if self.precision.last_gradient_norm is not None:
                    gradient_norms.append(self.precision.last_gradient_norm)
                if self.components.scheduler is not None:
                    self.components.scheduler.step()
                self.global_step += 1
            loss_total = loss_total + loss.total.detach()
            for name, value in loss.components.items():
                component_totals[name] = (
                    component_totals.get(name, torch.zeros((), device=self.device)) + value.detach()
                )
            if batch.targets is None:
                raise TrainingPreflightError("Training batch lost labeled targets.")
            correct_total = (
                correct_total
                + (_get_logits(weak_output).detach().argmax(dim=1) == batch.targets).sum()
            )
            observed_total += len(batch.sample_ids)
            step_count += 1
            if collect_loss and loss.per_sample_supervised is not None:
                detached_loss = loss.per_sample_supervised.detach().cpu().float()
                if detached_loss.shape != (len(batch.sample_ids),):
                    raise ValueError("per_sample_supervised must have shape [B].")
                for index, sample_id in enumerate(batch.sample_ids):
                    _store_once(per_sample_loss, sample_id, detached_loss[index])
            if track_sample_state:
                weak_logits = _get_logits(weak_output).detach()
                epoch_sample_ids.extend(batch.sample_ids)
                prediction_parts.append(weak_logits.argmax(dim=1))
            if collect_embeddings:
                embedding = _get_embedding(weak_output).detach().cpu().float()
                for index, sample_id in enumerate(batch.sample_ids):
                    _store_once(per_sample_embedding, sample_id, embedding[index])
            if collect_agreement:
                logits = _get_logits(weak_output).detach().cpu().float()
                strong_logits = (
                    None
                    if strong_output is None
                    else _get_logits(strong_output).detach().cpu().float()
                )
                for index, sample_id in enumerate(batch.sample_ids):
                    _store_once(per_sample_logits, sample_id, logits[index])
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
            if self.precision.last_gradient_norm is not None:
                gradient_norms.append(self.precision.last_gradient_norm)
            if self.components.scheduler is not None:
                self.components.scheduler.step()
            self.global_step += 1
        _synchronize_if_cuda(self.device)
        epoch_seconds = time.perf_counter() - epoch_started
        gradient_tensor = (
            torch.stack(gradient_norms).float()
            if gradient_norms
            else torch.zeros(1, device=self.device)
        )
        train_stats: dict[str, object] = {
            "loss_total": float((loss_total / max(1, step_count)).cpu().item()),
            "loss_components": {
                name: float((value / max(1, step_count)).cpu().item())
                for name, value in sorted(component_totals.items())
            },
            "train_top1": float((correct_total.float() / max(1, observed_total)).cpu().item()),
            "gradient_norm_mean": float(gradient_tensor.mean().cpu().item()),
            "gradient_norm_max": float(gradient_tensor.max().cpu().item()),
            "gradient_scaler": float(self.precision.scaler.get_scale()),
            "learning_rates": {
                str(group.get("name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.components.optimizer.param_groups)
            },
            "peak_gpu_memory_mib": (
                float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
                if self.device.type == "cuda"
                else 0.0
            ),
            "run_free_disk_gib": float(
                shutil.disk_usage(self.components.run_context.run_dir).free / (1024**3)
            ),
            "epoch_seconds": epoch_seconds,
            "samples_per_second": len(self.components.train_records) / max(epoch_seconds, 1e-12),
        }
        head = getattr(self.components.model, "head", None)
        current_temperature = getattr(head, "current_temperature", None)
        if callable(current_temperature):
            train_stats["temperature"] = float(current_temperature().detach().float().cpu().item())
        parameter_report = getattr(self.components.model, "trainable_parameter_report", None)
        if callable(parameter_report):
            train_stats["parameter_report"] = parameter_report()
        if not track_sample_state:
            return train_stats
        if len(set(epoch_sample_ids)) != len(epoch_sample_ids):
            raise ValueError("sample_id was seen more than once in the same epoch.")
        predictions = (
            torch.cat(prediction_parts).detach().cpu().tolist() if prediction_parts else []
        )
        per_sample_prediction = {
            sample_id: int(prediction)
            for sample_id, prediction in zip(epoch_sample_ids, predictions, strict=True)
        }
        missing = sorted(set(ordered_ids) - set(per_sample_prediction))
        extra = sorted(set(per_sample_prediction) - set(ordered_ids))
        if missing or extra:
            raise ValueError(
                f"Training epoch sample IDs mismatch: missing={missing}, extra={extra}."
            )
        train_stats.update(
            {
                "per_sample_loss": {
                    sample_id: per_sample_loss.get(sample_id, torch.tensor(0.0))
                    for sample_id in ordered_ids
                },
                "per_sample_prediction": {
                    sample_id: per_sample_prediction[sample_id] for sample_id in ordered_ids
                },
                "per_sample_logits": {
                    sample_id: per_sample_logits[sample_id]
                    for sample_id in ordered_ids
                    if sample_id in per_sample_logits
                },
                "per_sample_strong_logits": {
                    sample_id: per_sample_strong_logits[sample_id]
                    for sample_id in ordered_ids
                    if sample_id in per_sample_strong_logits
                },
                "per_sample_embedding": {
                    sample_id: per_sample_embedding[sample_id]
                    for sample_id in ordered_ids
                    if sample_id in per_sample_embedding
                },
            }
        )
        return train_stats

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
        predictions_by_id = _int_mapping(train_stats["per_sample_prediction"])
        ordered_predictions = torch.tensor(
            [predictions_by_id[record.sample_id] for record in records], dtype=torch.int64
        )
        history_updated = update_prediction_history_from_top1(
            previous_states,
            ordered_predictions,
            epoch=epoch,
            history_window=self.config.noise.signals.prediction_stability.window,
        )
        should_update_trust = self._trust_update_due(epoch)
        if not should_update_trust:
            return history_updated

        logits_by_id = _tensor_mapping(train_stats["per_sample_logits"])
        strong_logits_by_id = _tensor_mapping(train_stats["per_sample_strong_logits"])
        embeddings_by_id = _tensor_mapping(train_stats["per_sample_embedding"])
        losses_by_id = _tensor_mapping(train_stats["per_sample_loss"])
        trust_aggregator = self.components.trust_aggregator
        if trust_aggregator is None:  # pragma: no cover - narrowed by _trust_update_due.
            raise RuntimeError("noise update requires a trust aggregator.")
        raw_signals = self._raw_trust_signals(
            records=records,
            previous_states=previous_states,
            predictions_by_id=predictions_by_id,
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
        weights = self.config.noise.weights
        weighted = apply_supervision_weights(
            partitioned,
            trusted=weights.trusted,
            uncertain_min=weights.uncertain_min,
            uncertain_max=weights.uncertain_max,
            suspicious=weights.suspicious,
            epoch=epoch,
        )
        if self.components.curriculum is not None:
            return self.components.curriculum.apply(weighted, epoch)
        return weighted

    def _trust_update_due(self, epoch: int) -> bool:
        return (
            self.config.noise.enabled
            and self.components.trust_aggregator is not None
            and epoch >= self.config.noise.warmup_epochs
            and (epoch - self.config.noise.warmup_epochs)
            % self.config.noise.update_interval_epochs
            == 0
        )

    def _raw_trust_signals(
        self,
        *,
        records: list[SampleRecord],
        previous_states: list[SampleState],
        predictions_by_id: Mapping[str, int],
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
            window = self.config.noise.signals.prediction_stability.window
            values = []
            for record, state in zip(records, previous_states, strict=True):
                current = predictions_by_id[record.sample_id]
                values.append(
                    torch.tensor(
                        prediction_stability_from_history(
                            state.prediction_history,
                            current=current,
                            window=window,
                        )
                    )
                )
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

    def _restore_feature_cosine_history(self) -> None:
        if not self.config.evaluation.feature_drift_guard.enabled:
            return
        path = self.components.artifact_store.metric("epoch_metrics.jsonl")
        if not path.is_file():
            raise TrainingPreflightError("B2 resume lacks prior feature-cosine metrics.")
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise TrainingPreflightError("B2 resume has an empty epoch metric history.")
        latest = json.loads(rows[-1])
        value = latest.get("val/feature_cosine_to_base")
        if not isinstance(value, int | float):
            raise TrainingPreflightError("B2 resume lacks the latest feature cosine.")
        self.previous_feature_cosine = float(value)

    def _guard_feature_drift(self, result: EvaluationResult) -> None:
        """Record ordinary drift as a warning and stop only catastrophic changes."""

        guard = self.config.evaluation.feature_drift_guard
        if not guard.enabled:
            return
        value = result.metrics.get("val/feature_cosine_to_base")
        if value is None:
            raise NonFiniteTrainingError("Feature drift guard requires val/feature_cosine_to_base.")
        current = float(value)
        if not math.isfinite(current):
            raise NonFiniteTrainingError("Feature cosine is NaN or Inf.")
        epoch_drop = (
            None
            if self.previous_feature_cosine is None
            else self.previous_feature_cosine - current
        )
        result.metrics["val/feature_cosine_epoch_drop"] = epoch_drop

        catastrophic_reasons: list[str] = []
        if current < guard.catastrophic_minimum_cosine:
            catastrophic_reasons.append(
                f"raw cosine {current:.6f} is below catastrophic floor "
                f"{guard.catastrophic_minimum_cosine:.6f}"
            )
        if epoch_drop is not None and epoch_drop > guard.catastrophic_maximum_epoch_drop:
            catastrophic_reasons.append(
                f"raw cosine dropped by {epoch_drop:.6f}, above catastrophic limit "
                f"{guard.catastrophic_maximum_epoch_drop:.6f}"
            )
        if catastrophic_reasons:
            _write_json(
                self.components.artifact_store.metric("feature_drift_failure.json"),
                {
                    "current_raw_cosine": current,
                    "previous_raw_cosine": self.previous_feature_cosine,
                    "epoch_drop": epoch_drop,
                    "reasons": catastrophic_reasons,
                },
                overwrite=True,
            )
            raise NonFiniteTrainingError(
                "Catastrophic feature drift: " + "; ".join(catastrophic_reasons) + "."
            )

        warning_reasons: list[str] = []
        if current < guard.minimum_cosine:
            warning_reasons.append(
                f"raw cosine {current:.6f} is below diagnostic floor "
                f"{guard.minimum_cosine:.6f}"
            )
        if epoch_drop is not None and epoch_drop > guard.maximum_epoch_drop:
            warning_reasons.append(
                f"raw cosine dropped by {epoch_drop:.6f}, above diagnostic limit "
                f"{guard.maximum_epoch_drop:.6f}"
            )
        result.metrics["val/feature_drift_warning"] = float(bool(warning_reasons))
        if warning_reasons:
            message = "; ".join(warning_reasons)
            result.metric_reasons["val/feature_drift_warning"] = message
            warnings.warn(f"Feature drift warning: {message}.", RuntimeWarning, stacklevel=2)
        self.previous_feature_cosine = current

    def _export_final_model(self) -> Path | None:
        destination = self.components.artifact_store.artifact("model.pt")
        export = getattr(self.components.model, "export_single_model", None)
        if callable(export):
            before_logits: Tensor | None = None
            equivalence_batch: Batch | None = None
            if _trainability_stage(self.config) == "B2":
                equivalence_batch = _first_device_batch(self.components.val_loader, self.device)
                self.components.model.eval()
                with torch.inference_mode(), self.precision.autocast():
                    before_logits = (
                        _get_logits(
                            _forward_batch(self.components.model, equivalence_batch, strong=False)
                        )
                        .detach()
                        .float()
                    )
            if self.components.clip_weight_metadata is None:
                exported = Path(export(destination))
            else:
                exported = Path(
                    export(
                        destination,
                        preprocessing_spec=self.components.preprocessing_spec,
                        config_summary=self.components.config_summary,
                        class_to_idx=self.components.run_context.class_to_idx,
                        mapping_digest=mapping_digest(self.components.run_context.class_to_idx),
                        clip_weight_metadata=self.components.clip_weight_metadata,
                    )
                )
            if before_logits is not None and equivalence_batch is not None:
                self._write_lora_merge_equivalence(
                    before_logits, equivalence_batch, exported_model=exported
                )
            return exported
        return None

    def _write_lora_merge_equivalence(
        self,
        before_logits: Tensor,
        batch: Batch,
        *,
        exported_model: Path,
    ) -> None:
        self.components.model.eval()
        with torch.inference_mode(), self.precision.autocast():
            after_logits = (
                _get_logits(_forward_batch(self.components.model, batch, strong=False))
                .detach()
                .float()
            )
        if before_logits.shape != after_logits.shape:
            raise NonFiniteTrainingError("LoRA merge changed the validation logit shape.")
        max_abs = float((before_logits - after_logits).abs().max().cpu().item())
        prediction_mismatches = int(
            (before_logits.argmax(dim=1) != after_logits.argmax(dim=1)).sum().cpu().item()
        )
        tolerance = self.config.evaluation.lora_merge_atol
        passed = max_abs <= tolerance and prediction_mismatches == 0
        package = load_export_package(exported_model)
        exported_state = package.get("model_state")
        artifact_valid = isinstance(exported_state, Mapping) and all(
            ".lora_" not in str(key) for key in exported_state
        )
        passed = passed and artifact_valid
        _write_json(
            self.components.artifact_store.metric("lora_merge_equivalence.json"),
            {
                "batch_size": len(batch.sample_ids),
                "max_absolute_logit_error": max_abs,
                "prediction_mismatch_count": prediction_mismatches,
                "sample_ids": list(batch.sample_ids),
                "tolerance": tolerance,
                "export_artifact_validated": artifact_valid,
                "valid": passed,
            },
            overwrite=False,
        )
        if not passed:
            raise NonFiniteTrainingError(
                "LoRA merge output equivalence failed: "
                f"max_abs={max_abs:.6g}, mismatches={prediction_mismatches}."
            )

    def _save_final_prototypes(self) -> None:
        if not self.config.model.head.prototype_init.enabled:
            return
        head = getattr(self.components.model, "head", None)
        weight = getattr(head, "weight", None)
        if not isinstance(weight, Tensor):
            linear = getattr(head, "linear", None)
            weight = getattr(linear, "weight", None)
        if not isinstance(weight, Tensor):
            raise TrainingPreflightError("Prototype-enabled head does not expose weights.")
        atomic_save_with_writer(
            self.components.artifact_store.artifact("final_prototypes.pt"),
            lambda temporary: torch.save(weight.detach().cpu().float().contiguous(), temporary),
            overwrite=False,
        )

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
    """Require gradients to exist only on the stage-authorized parameters.

    Args:
        model: Student model after backward.
        stage: Stage name controlling freeze policy.

    Raises:
        NonFiniteTrainingError: If any forbidden parameter receives a gradient.
    """

    normalized_stage = "B0" if stage == "init" else stage
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if normalized_stage in {"B0", "B1"}:
            authorized = not name.startswith("backbone.")
        elif normalized_stage == "B2":
            authorized = name.startswith("head.") or ".lora_" in f".{name}"
        else:
            authorized = True
        if not authorized:
            raise NonFiniteTrainingError(
                f"Unauthorized {normalized_stage} gradient was created: {name}."
            )


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


def _teacher_embedding(teacher: Any | None, batch: Batch) -> Tensor | None:
    if teacher is None:
        return None
    with torch.no_grad():
        if batch.image_weak is None:
            raise ValueError("Teacher path requires image tensors, not cached features.")
        return teacher.encode_image(batch.image_weak)


def _forward_batch(model: Any, batch: Batch, *, strong: bool) -> ModelOutput:
    embedding = batch.embedding_strong if strong else batch.embedding_weak
    if embedding is not None:
        forward_embeddings = getattr(model, "forward_embeddings", None)
        if not callable(forward_embeddings):
            raise ValueError("Cached feature batch requires model.forward_embeddings().")
        return cast(ModelOutput, forward_embeddings(embedding))
    images = batch.image_strong if strong else batch.image_weak
    if images is None:
        raise ValueError("Batch contains neither images nor cached embeddings.")
    return cast(ModelOutput, model(images))


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


def _int_mapping(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise TypeError("train stat prediction mapping must be a mapping.")
    result: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("train stat predictions must map string IDs to integers.")
        if value < 0:
            raise ValueError("train stat predictions must be non-negative.")
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
    components = train_stats.get("loss_components")
    if isinstance(components, Mapping):
        for name, value in components.items():
            metric_name = str(name)
            record[f"train/{metric_name}"] = _float_stat(value)
    for source, destination in (
        ("train_top1", "train/top1"),
        ("gradient_norm_mean", "optimizer/gradient_norm_mean"),
        ("gradient_norm_max", "optimizer/gradient_norm_max"),
        ("gradient_scaler", "optimizer/gradient_scaler"),
        ("temperature", "model/temperature"),
        ("peak_gpu_memory_mib", "system/max_gpu_memory_mib"),
        ("run_free_disk_gib", "system/run_free_disk_gib"),
    ):
        if source in train_stats:
            record[destination] = _float_stat(train_stats[source])
    learning_rates = train_stats.get("learning_rates")
    if isinstance(learning_rates, Mapping):
        for name, value in learning_rates.items():
            record[f"optimizer/lr_{name}"] = _float_stat(value)
    parameter_report = train_stats.get("parameter_report")
    if isinstance(parameter_report, Mapping):
        for name in ("trainable_parameters", "trainable_ratio", "lora_trainable_parameters"):
            if name in parameter_report:
                record[f"model/{name}"] = _float_stat(parameter_report[name])
    for key in ("epoch_seconds", "samples_per_second", "validation_seconds"):
        if key in train_stats:
            record[f"timing/{key}"] = _float_stat(train_stats[key])
    record.update(val_result.metrics)
    train_top1 = record.get("train/top1")
    val_top1 = record.get("val/top1")
    if isinstance(train_top1, int | float) and isinstance(val_top1, int | float):
        record["diagnostic/train_val_top1_gap"] = float(train_top1 - val_top1)
    for name, reason in val_result.metric_reasons.items():
        record[f"{name}/reason"] = reason
    return record


def _first_device_batch(loader: Iterable[Batch], device: torch.device) -> Batch:
    try:
        return next(iter(BatchDeviceIterator(loader, device)))
    except StopIteration as exc:
        raise TrainingPreflightError("Validation loader produced no equivalence batch.") from exc


def _write_json(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )


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


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _best_checkpoint_name(metric_name: str) -> str:
    leaf = metric_name.rsplit("/", 1)[-1]
    safe = "".join(
        character if character.isalnum() or character == "_" else "_" for character in leaf
    )
    return f"best_{safe}.pt"
