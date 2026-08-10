"""Configuration-driven assembly for real B0-B6 training and evaluation."""

from __future__ import annotations

import copy
import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from noisyclip.config.loader import load_config, write_resolved_config
from noisyclip.config.schema import ProjectConfig
from noisyclip.data.dataset import ManifestImageDataset, collate_batch
from noisyclip.data.feature_cache import (
    FrozenFeatureLoader,
    ReferenceFeatureCache,
    build_frozen_feature_cache,
    feature_cache_signature,
    load_feature_tensor,
    load_reference_feature_cache,
    reference_feature_cache_signature,
)
from noisyclip.data.image_io import build_image_loader
from noisyclip.data.leakage import check_root_boundaries
from noisyclip.data.manifests import data_digest, read_manifest
from noisyclip.data.records import Batch
from noisyclip.data.sampler import build_sampler
from noisyclip.data.transforms import build_transform
from noisyclip.engine.checkpoint import load_checkpoint
from noisyclip.engine.context import RunContext
from noisyclip.engine.device import move_batch_to_device
from noisyclip.engine.evaluator import EvaluationResult, Evaluator, save_evaluation_artifacts
from noisyclip.engine.seed import set_seed
from noisyclip.engine.trainer import Trainer, TrainerComponents, TrainResult
from noisyclip.losses.composite import RobustCompositeLoss
from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import build_classifier_head
from noisyclip.models.clip_loader import ClipBackend, load_clip_vit_b32
from noisyclip.models.lora import LoraInjectionConfig, inject_lora_into_visual_transformer
from noisyclip.models.prototypes import build_prototype_builder
from noisyclip.models.student import NoisyCLIPStudent
from noisyclip.models.teacher import FrozenTeacherModel
from noisyclip.noise.curriculum import PartitionCurriculum
from noisyclip.noise.state import JsonSampleStateStore
from noisyclip.noise.trust import ClasswiseTrustAggregator
from noisyclip.submission.mapping import ClassMapping, load_class_mapping
from noisyclip.tracking.artifacts import ArtifactStore
from noisyclip.tracking.environment import collect_environment_snapshot
from noisyclip.tracking.manifest import RunManifest
from noisyclip.utils.atomic import atomic_save_with_writer, atomic_write_bytes, ensure_free_space
from noisyclip.utils.hashing import stable_hash


class AssemblyError(ValueError):
    """Raised when validated configuration cannot form a legal runtime."""


def run_training(
    config_path: Path | str,
    *,
    run_id: str,
    resume_checkpoint: Path | str | None = None,
    clip_backend: ClipBackend | None = None,
) -> TrainResult:
    """Assemble and execute one configuration-driven training run."""

    config = load_config(config_path)
    trainer = assemble_trainer(
        config,
        run_id=run_id,
        resume_checkpoint=resume_checkpoint,
        clip_backend=clip_backend,
    )
    return trainer.fit()


def assemble_trainer(
    config: ProjectConfig,
    *,
    run_id: str,
    resume_checkpoint: Path | str | None = None,
    clip_backend: ClipBackend | None = None,
) -> Trainer:
    """Build datasets, official CLIP student, optimization, and run tracking."""

    _reject_unintegrated_upper_bound_modules(config)
    paths = _training_paths(config)
    mapping = load_class_mapping(paths["class_mapping"])
    if mapping.num_classes != config.data.expected_num_classes:
        raise AssemblyError(
            f"Class mapping has {mapping.num_classes} classes; "
            f"config expects {config.data.expected_num_classes}."
        )
    train_records = read_manifest(paths["train_manifest"])
    val_records = read_manifest(paths["val_manifest"])
    _validate_fixed_splits(train_records, val_records, mapping)
    digest = data_digest(train_records + val_records, dict(mapping.class_to_idx))
    config_digest = stable_hash(config.model_dump(mode="json"))
    device = _resolve_device(config.trainer.device)
    cache_root = _concrete_path(config.paths.cache_root, "paths.cache_root")
    run_root = _concrete_path(config.paths.run_root, "paths.run_root")
    ensure_free_space(run_root, int(config.tracking.minimum_free_disk_gib * 1024**3))

    resume_path = None if resume_checkpoint is None else Path(resume_checkpoint).resolve()
    if resume_path is None:
        run_manifest = RunManifest.create(
            run_root=run_root,
            run_id=run_id,
            metadata={
                "config_digest": config_digest,
                "data_digest": digest,
                "class_mapping_digest": mapping.digest,
                "seed": config.experiment.seed,
            },
            fail_if_run_exists=config.tracking.fail_if_run_exists,
        )
    else:
        run_dir = (run_root / run_id).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        run_manifest = RunManifest.open(run_dir)
        if run_manifest.status == "COMPLETED":
            raise AssemblyError("A completed run cannot be resumed in place.")

    artifact_store = ArtifactStore(run_manifest.run_dir)
    _snapshot_run_inputs(
        config,
        artifact_store,
        paths=paths,
        config_digest=config_digest,
        data_digest_value=digest,
        mapping=mapping,
        resume=resume_path is not None,
    )

    set_seed(config.experiment.seed, deterministic=config.trainer.deterministic)
    try:
        loaded = load_clip_vit_b32(
            model_name=config.model.backbone.name,
            pretrained=config.model.backbone.pretrained,
            device=device,
            cache_dir=cache_root,
            backend=clip_backend,
        )
        model, teacher = _build_model(config, mapping, loaded.model, device)
        reference_cache = _load_reference_cache(
            config,
            data_digest_value=digest,
            class_mapping_digest=mapping.digest,
            clip_weight_sha256=_clip_identity(loaded.metadata),
            train_records=train_records,
            val_records=val_records,
        )
        train_loader: Iterable[Batch]
        val_loader: Iterable[Batch]
        initial_prototypes: Tensor | None = None
        if config.trainer.frozen_feature_cache.enabled:
            feature_root = _concrete_path(
                str(config.trainer.frozen_feature_cache.directory),
                "trainer.frozen_feature_cache.directory",
            )
            signature = feature_cache_signature(
                config,
                data_digest=digest,
                class_mapping_digest=mapping.digest,
                clip_weight_sha256=_clip_identity(loaded.metadata),
            )
            train_loader = FrozenFeatureLoader(
                feature_root,
                split="train",
                batch_size=config.trainer.batch_size,
                seed=config.experiment.seed,
                expected_signature=signature,
                verify_hashes=config.trainer.frozen_feature_cache.verify_hashes,
            )
            val_loader = FrozenFeatureLoader(
                feature_root,
                split="val",
                batch_size=config.trainer.batch_size,
                seed=config.experiment.seed,
                expected_signature=signature,
                verify_hashes=False,
            )
            if config.model.head.prototype_init.enabled and resume_path is None:
                initial_prototypes = _initialize_head_from_feature_cache(
                    config, model, feature_root, train_loader
                )
        else:
            train_loader, val_loader = _build_loaders(config, train_records, val_records, paths)
            if config.model.head.prototype_init.enabled and resume_path is None:
                if reference_cache is not None:
                    initial_prototypes = _initialize_head_from_reference_cache(
                        config, model, reference_cache
                    )
                else:
                    initial_prototypes = _initialize_head_from_prototypes(
                        config,
                        model,
                        train_records,
                        paths["train_root"],
                        device,
                    )
        if resume_path is None:
            _write_model_audit(
                config,
                model,
                artifact_store,
                reference_cache=reference_cache,
                initial_prototypes=initial_prototypes,
            )
        optimizer = _build_optimizer(config, model)
        scheduler = _build_scheduler(config, optimizer, len(train_loader))
        loss = RobustCompositeLoss(config.loss)
        state_store = JsonSampleStateStore(
            artifact_store.sample_state_dir(),
            expected_sample_ids=[record.sample_id for record in train_records],
        )
        trust = ClasswiseTrustAggregator.from_config(config.noise) if config.noise.enabled else None
        curriculum = (
            PartitionCurriculum.from_config(config.noise.curriculum)
            if config.noise.curriculum.enabled
            else None
        )
        context = RunContext(
            run_id=run_id,
            run_dir=run_manifest.run_dir,
            seed=config.experiment.seed,
            num_classes=mapping.num_classes,
            class_to_idx=mapping.class_to_idx,
            config_digest=config_digest,
            data_digest=digest,
        )
        components = TrainerComponents(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            train_records=train_records,
            state_store=state_store,
            run_context=context,
            artifact_store=artifact_store,
            run_manifest=run_manifest,
            teacher=teacher,
            trust_aggregator=trust,
            curriculum=curriculum,
            clip_weight_metadata=asdict(loaded.metadata),
            preprocessing_spec=_preprocessing_spec(config),
            config_summary=config.model_dump(mode="json"),
            resume_checkpoint=resume_path,
            base_val_embeddings=(
                None if reference_cache is None else reference_cache.val_by_sample
            ),
            reference_cache_signature=(
                None if reference_cache is None else reference_cache.signature
            ),
        )
        return Trainer(config=config, components=components, device=device)
    except Exception as exc:
        run_manifest.mark_failed(str(exc), stage="assembly")
        raise


def evaluate_checkpoint(
    run_dir: Path | str,
    checkpoint: Path | str,
    *,
    clip_backend: ClipBackend | None = None,
) -> EvaluationResult:
    """Rebuild one run model, load a checkpoint, and evaluate its fixed val split."""

    root = Path(run_dir).resolve()
    config = load_config(root / "resolved_config.yaml")
    mapping = load_class_mapping(root / "data" / "class_to_idx.json")
    train_records = read_manifest(root / "data" / "train_manifest.json")
    val_records = read_manifest(root / "data" / "val_manifest.json")
    digest = data_digest(train_records + val_records, dict(mapping.class_to_idx))
    config_digest = stable_hash(config.model_dump(mode="json"))
    device = _resolve_device(config.trainer.device)
    loaded = load_clip_vit_b32(
        model_name=config.model.backbone.name,
        pretrained=config.model.backbone.pretrained,
        device=device,
        cache_dir=_concrete_path(config.paths.cache_root, "paths.cache_root"),
        backend=clip_backend,
    )
    model, _ = _build_model(config, mapping, loaded.model, device, build_teacher=False)
    metadata = load_checkpoint(checkpoint, model=model, map_location=device)
    if metadata.config_digest != config_digest or metadata.data_digest != digest:
        raise AssemblyError("Checkpoint identity does not match resolved config and fixed data.")
    loader: Iterable[Batch]
    if config.trainer.frozen_feature_cache.enabled:
        feature_root = _concrete_path(
            str(config.trainer.frozen_feature_cache.directory),
            "trainer.frozen_feature_cache.directory",
        )
        signature = feature_cache_signature(
            config,
            data_digest=digest,
            class_mapping_digest=mapping.digest,
            clip_weight_sha256=_clip_identity(loaded.metadata),
        )
        loader = FrozenFeatureLoader(
            feature_root,
            split="val",
            batch_size=config.trainer.batch_size,
            seed=config.experiment.seed,
            expected_signature=signature,
            verify_hashes=config.trainer.frozen_feature_cache.verify_hashes,
        )
    else:
        val_dataset = ManifestImageDataset(
            val_records,
            data_root=_concrete_path(config.paths.train_root, "paths.train_root"),
            split="val",
            image_weak_transform=build_transform(
                config.data,
                split="val",
                normalize=not config.data.normalize_on_device,
            ),
            image_loader=build_image_loader(config.data.decode_backend),
        )
        loader = DataLoader(
            val_dataset,
            batch_size=config.trainer.batch_size,
            shuffle=False,
            num_workers=config.trainer.num_workers,
            pin_memory=config.trainer.pin_memory and device.type == "cuda",
            collate_fn=collate_batch,
            prefetch_factor=(
                config.trainer.prefetch_factor if config.trainer.num_workers > 0 else None
            ),
            persistent_workers=(
                config.trainer.persistent_workers if config.trainer.num_workers > 0 else False
            ),
        )
    evaluator = Evaluator(
        model=model,
        num_classes=mapping.num_classes,
        device=device,
        runtime_tensor_checks=config.trainer.runtime_tensor_checks,
    )
    reference_cache = _load_reference_cache(
        config,
        data_digest_value=digest,
        class_mapping_digest=mapping.digest,
        clip_weight_sha256=_clip_identity(loaded.metadata),
        train_records=train_records,
        val_records=val_records,
    )
    result: EvaluationResult = evaluator.evaluate(
        loader,
        base_embeddings=None if reference_cache is None else reference_cache.val_by_sample,
    )
    output_dir = root / "metrics" / f"evaluation_epoch_{metadata.epoch:04d}"
    save_evaluation_artifacts(result, output_dir)
    atomic_write_bytes(
        output_dir / "metrics.json",
        (json.dumps(result.metrics, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return result


def build_feature_cache_from_config(
    config_path: Path | str,
    *,
    clip_backend: ClipBackend | None = None,
) -> Path:
    """Build a required B0/B1 feature cache without creating a training run."""

    config = load_config(config_path)
    if not config.trainer.frozen_feature_cache.enabled:
        raise AssemblyError(
            "Feature-cache build requires trainer.frozen_feature_cache.enabled=true."
        )
    paths = _training_paths(config)
    mapping = load_class_mapping(paths["class_mapping"])
    train_records = read_manifest(paths["train_manifest"])
    val_records = read_manifest(paths["val_manifest"])
    _validate_fixed_splits(train_records, val_records, mapping)
    digest = data_digest(train_records + val_records, dict(mapping.class_to_idx))
    device = _resolve_device(config.trainer.device)
    set_seed(config.experiment.seed, deterministic=config.trainer.deterministic)
    loaded = load_clip_vit_b32(
        model_name=config.model.backbone.name,
        pretrained=config.model.backbone.pretrained,
        device=device,
        cache_dir=_concrete_path(config.paths.cache_root, "paths.cache_root"),
        backend=clip_backend,
    )
    model, _ = _build_model(config, mapping, loaded.model, device, build_teacher=False)
    destination = _concrete_path(
        str(config.trainer.frozen_feature_cache.directory),
        "trainer.frozen_feature_cache.directory",
    )
    return build_frozen_feature_cache(
        destination,
        config=config,
        model=model,
        train_records=train_records,
        val_records=val_records,
        train_root=paths["train_root"],
        device=device,
        data_digest=digest,
        class_mapping_digest=mapping.digest,
        clip_weight_sha256=_clip_identity(loaded.metadata),
    )


def _training_paths(config: ProjectConfig) -> dict[str, Path]:
    train_root = _concrete_path(config.paths.train_root, "paths.train_root")
    test_root = _concrete_path(config.paths.test_root, "paths.test_root")
    run_root = _concrete_path(config.paths.run_root, "paths.run_root")
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training root does not exist: {train_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root does not exist: {test_root}")
    root_issue = check_root_boundaries(train_root, test_root)
    if root_issue:
        raise AssemblyError(root_issue)
    audit_root = run_root / "data"
    return {
        "train_root": train_root,
        "test_root": test_root,
        "run_root": run_root,
        "class_mapping": _optional_path(
            config.paths.class_mapping, audit_root / "class_to_idx.json"
        ),
        "train_manifest": _optional_path(
            config.paths.train_manifest, audit_root / "train_manifest.json"
        ),
        "val_manifest": _optional_path(config.paths.val_manifest, audit_root / "val_manifest.json"),
    }


def _validate_fixed_splits(
    train_records: list[Any], val_records: list[Any], mapping: ClassMapping
) -> None:
    if not train_records or not val_records:
        raise AssemblyError("Fixed train and validation manifests must both be non-empty.")
    if any(record.split != "train" for record in train_records):
        raise AssemblyError("train_manifest contains a non-train record.")
    if any(record.split != "val" for record in val_records):
        raise AssemblyError("val_manifest contains a non-val record.")
    valid_pairs = set(mapping.class_to_idx.items())
    for record in train_records + val_records:
        if (record.class_id, record.target) not in valid_pairs:
            raise AssemblyError(f"Manifest label disagrees with class mapping: {record.sample_id}")
    train_ids = {record.sample_id for record in train_records}
    val_ids = {record.sample_id for record in val_records}
    if train_ids & val_ids:
        raise AssemblyError("Train and validation manifests share sample IDs.")


def _build_model(
    config: ProjectConfig,
    mapping: ClassMapping,
    clip_model: nn.Module,
    device: torch.device,
    *,
    build_teacher: bool = True,
) -> tuple[NoisyCLIPStudent, FrozenTeacherModel | None]:
    backbone = CLIPImageBackbone(clip_model, freeze=True)
    teacher = None
    if build_teacher and config.model.teacher.enabled:
        teacher_backbone = copy.deepcopy(backbone).to(device)
        teacher = FrozenTeacherModel(
            teacher_backbone,
            embedding_dim=backbone.embedding_dim,
        )
    stage: Literal["B0", "B1", "B2"] = (
        "B2"
        if config.model.lora.enabled
        else ("B1" if config.model.head.type == "cosine" else "B0")
    )
    if config.model.lora.enabled:
        inject_lora_into_visual_transformer(
            backbone.clip_model,
            LoraInjectionConfig(
                target_blocks=config.model.lora.target_blocks,
                target_projections=config.model.lora.target_projections,
                rank=config.model.lora.rank,
                alpha=config.model.lora.alpha,
                dropout=config.model.lora.dropout,
            ),
        )
    head = build_classifier_head(
        head_type=config.model.head.type,
        embedding_dim=backbone.embedding_dim,
        num_classes=mapping.num_classes,
        temperature_init=config.model.head.temperature_init,
        temperature_min=config.model.head.temperature_min,
        temperature_max=config.model.head.temperature_max,
    )
    student = NoisyCLIPStudent(backbone=backbone, head=head, stage=stage)
    student.to(device)
    return student, teacher


def _build_loaders(
    config: ProjectConfig,
    train_records: list[Any],
    val_records: list[Any],
    paths: dict[str, Path],
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset = ManifestImageDataset(
        train_records,
        data_root=paths["train_root"],
        split="train",
        image_weak_transform=build_transform(
            config.data,
            split="train",
            seed=config.experiment.seed,
            normalize=not config.data.normalize_on_device,
        ),
        image_strong_transform=(
            build_transform(
                config.data,
                split="train",
                strong=True,
                seed=config.experiment.seed,
                normalize=not config.data.normalize_on_device,
            )
            if config.data.strong_transform.enabled
            else None
        ),
        training=True,
        image_loader=build_image_loader(config.data.decode_backend),
    )
    val_dataset = ManifestImageDataset(
        val_records,
        data_root=paths["train_root"],
        split="val",
        image_weak_transform=build_transform(
            config.data, split="val", normalize=not config.data.normalize_on_device
        ),
        image_loader=build_image_loader(config.data.decode_backend),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        sampler=build_sampler(train_dataset.records, shuffle=True, seed=config.experiment.seed),
        num_workers=config.trainer.num_workers,
        pin_memory=config.trainer.pin_memory and config.trainer.device.startswith("cuda"),
        collate_fn=collate_batch,
        prefetch_factor=(
            config.trainer.prefetch_factor if config.trainer.num_workers > 0 else None
        ),
        persistent_workers=(
            config.trainer.persistent_workers if config.trainer.num_workers > 0 else False
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        sampler=build_sampler(val_dataset.records, shuffle=False, seed=config.experiment.seed),
        num_workers=config.trainer.num_workers,
        pin_memory=config.trainer.pin_memory and config.trainer.device.startswith("cuda"),
        collate_fn=collate_batch,
        prefetch_factor=(
            config.trainer.prefetch_factor if config.trainer.num_workers > 0 else None
        ),
        persistent_workers=(
            config.trainer.persistent_workers if config.trainer.num_workers > 0 else False
        ),
    )
    return train_loader, val_loader


@torch.inference_mode()
def _initialize_head_from_prototypes(
    config: ProjectConfig,
    model: NoisyCLIPStudent,
    train_records: list[Any],
    train_root: Path,
    device: torch.device,
) -> Tensor:
    method = config.model.head.prototype_init.method
    if method == "multi_prototype":
        raise AssemblyError("U3 multi-prototype initialization is not part of the B0-B6 head.")
    dataset = ManifestImageDataset(
        train_records,
        data_root=train_root,
        split="train",
        image_weak_transform=build_transform(
            config.data, split="val", normalize=not config.data.normalize_on_device
        ),
        image_loader=build_image_loader(config.data.decode_backend),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        collate_fn=collate_batch,
        pin_memory=config.trainer.pin_memory and config.trainer.device.startswith("cuda"),
        prefetch_factor=(
            config.trainer.prefetch_factor if config.trainer.num_workers > 0 else None
        ),
        persistent_workers=(
            config.trainer.persistent_workers if config.trainer.num_workers > 0 else False
        ),
    )
    embeddings: list[Tensor] = []
    targets: list[Tensor] = []
    model.backbone.eval()
    for batch in loader:
        if batch.targets is None:
            raise AssemblyError("Prototype initialization requires labeled training data.")
        batch = move_batch_to_device(batch, device, non_blocking=True)
        if batch.image_weak is None or batch.targets is None:
            raise AssemblyError("Prototype initialization lost images or labeled targets.")
        embeddings.append(model.backbone.encode_image(batch.image_weak).cpu())
        targets.append(batch.targets.cpu())
    prototypes = build_prototype_builder(
        method,
        keep_fraction=config.model.head.prototype_init.keep_fraction,
    ).fit(
        torch.cat(embeddings),
        torch.cat(targets),
        None,
        config.data.expected_num_classes,
    )
    _copy_prototypes_to_head(model, prototypes)
    return prototypes.detach().cpu().float()


@torch.inference_mode()
def _initialize_head_from_feature_cache(
    config: ProjectConfig,
    model: NoisyCLIPStudent,
    feature_root: Path,
    loader: FrozenFeatureLoader,
) -> Tensor:
    features = load_feature_tensor(feature_root / "train_eval.pt")
    targets = torch.tensor(loader.metadata["train_targets"], dtype=torch.int64)
    method = config.model.head.prototype_init.method
    if method == "multi_prototype":
        raise AssemblyError("Multi-prototype cache initialization is not supported.")
    prototypes = build_prototype_builder(
        method,
        keep_fraction=config.model.head.prototype_init.keep_fraction,
    ).fit(features, targets, None, config.data.expected_num_classes)
    _copy_prototypes_to_head(model, prototypes)
    return prototypes.detach().cpu().float()


@torch.inference_mode()
def _initialize_head_from_reference_cache(
    config: ProjectConfig,
    model: NoisyCLIPStudent,
    reference: ReferenceFeatureCache,
) -> Tensor:
    method = config.model.head.prototype_init.method
    if method == "multi_prototype":
        raise AssemblyError("Multi-prototype reference initialization is not supported.")
    prototypes = build_prototype_builder(
        method,
        keep_fraction=config.model.head.prototype_init.keep_fraction,
    ).fit(
        reference.train_eval,
        reference.train_targets,
        None,
        config.data.expected_num_classes,
    )
    _copy_prototypes_to_head(model, prototypes)
    return prototypes.detach().cpu().float()


def _copy_prototypes_to_head(model: NoisyCLIPStudent, prototypes: Tensor) -> None:
    if hasattr(model.head, "weight"):
        model.head.weight.copy_(prototypes.to(model.head.weight.device))
    elif hasattr(model.head, "linear"):
        model.head.linear.weight.copy_(prototypes.to(model.head.linear.weight.device))
        model.head.linear.bias.zero_()
    else:  # pragma: no cover - protected by configured head allowlist.
        raise AssemblyError("Configured classifier head cannot accept prototype weights.")


def _load_reference_cache(
    config: ProjectConfig,
    *,
    data_digest_value: str,
    class_mapping_digest: str,
    clip_weight_sha256: str,
    train_records: list[Any],
    val_records: list[Any],
) -> ReferenceFeatureCache | None:
    cache = config.trainer.reference_feature_cache
    if not cache.enabled:
        return None
    root = _concrete_path(str(cache.directory), "trainer.reference_feature_cache.directory")
    signature = reference_feature_cache_signature(
        root,
        config,
        data_digest=data_digest_value,
        class_mapping_digest=class_mapping_digest,
        clip_weight_sha256=clip_weight_sha256,
    )
    reference = load_reference_feature_cache(
        root,
        expected_signature=signature,
        verify_hashes=cache.verify_hashes,
    )
    ordered_train = sorted(train_records, key=lambda row: (row.relative_path, row.sample_id))
    ordered_val = sorted(val_records, key=lambda row: (row.relative_path, row.sample_id))
    expected_train_ids = tuple(record.sample_id for record in ordered_train)
    expected_train_targets = torch.tensor(
        [record.target for record in ordered_train], dtype=torch.int64
    )
    expected_val_ids = tuple(record.sample_id for record in ordered_val)
    if reference.train_sample_ids != expected_train_ids:
        raise AssemblyError("Reference cache train sample IDs do not match the fixed split.")
    if not torch.equal(reference.train_targets, expected_train_targets):
        raise AssemblyError("Reference cache train targets do not match the fixed split.")
    if reference.val_sample_ids != expected_val_ids:
        raise AssemblyError("Reference cache val sample IDs do not match the fixed split.")
    return reference


def _write_model_audit(
    config: ProjectConfig,
    model: NoisyCLIPStudent,
    store: ArtifactStore,
    *,
    reference_cache: ReferenceFeatureCache | None,
    initial_prototypes: Tensor | None,
) -> None:
    trainable = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    report = {
        **model.trainable_parameter_report(),
        "stage": model.stage,
        "configured_precision": config.trainer.precision,
        "configured_precision_scope": "autocast_and_gradient_scaler",
        "backbone_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.backbone.parameters()}
        ),
        "head_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.head.parameters()}
        ),
        "trainable_parameters_detail": trainable,
        "reference_cache_signature": (
            None if reference_cache is None else reference_cache.signature
        ),
    }
    report["all_model_parameters_fp32"] = all(
        parameter.dtype == torch.float32 for parameter in model.parameters()
    )
    _write_json(store.metric("parameter_audit.json"), report, overwrite=False)
    if initial_prototypes is None:
        return
    _save_tensor(store.artifact("initial_prototypes.pt"), initial_prototypes)
    norms = torch.linalg.vector_norm(initial_prototypes.float(), dim=1)
    _write_json(
        store.artifact("prototype_initialization.json"),
        {
            "method": config.model.head.prototype_init.method,
            "keep_fraction": config.model.head.prototype_init.keep_fraction,
            "class_count": int(initial_prototypes.shape[0]),
            "embedding_dim": int(initial_prototypes.shape[1]),
            "minimum_norm": float(norms.min().item()),
            "maximum_norm": float(norms.max().item()),
            "finite": bool(torch.isfinite(initial_prototypes).all().item()),
            "reference_cache_signature": (
                None if reference_cache is None else reference_cache.signature
            ),
        },
        overwrite=False,
    )


def _save_tensor(path: Path, tensor: Tensor) -> None:
    atomic_save_with_writer(
        path,
        lambda temporary: torch.save(tensor.detach().cpu().contiguous(), temporary),
        overwrite=False,
    )


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )


def _clip_identity(metadata: Any) -> str:
    sha256 = getattr(metadata, "sha256", None)
    if isinstance(sha256, str) and sha256:
        return sha256
    return stable_hash(asdict(metadata))


def _build_optimizer(config: ProjectConfig, model: nn.Module) -> torch.optim.Optimizer:
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    lora = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and ".lora_" in f".{name}"
    ]
    groups: list[dict[str, Any]] = [
        {"name": "head", "params": head, "lr": config.trainer.optimizer.head_lr}
    ]
    if lora:
        groups.append({"name": "lora", "params": lora, "lr": config.trainer.optimizer.lora_lr})
    return torch.optim.AdamW(groups, weight_decay=config.trainer.optimizer.weight_decay)


def _build_scheduler(
    config: ProjectConfig,
    optimizer: torch.optim.Optimizer,
    microbatches_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = math.ceil(microbatches_per_epoch / config.trainer.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * config.trainer.epochs)
    warmup_steps = steps_per_epoch * config.trainer.scheduler.warmup_epochs
    minimum = config.trainer.scheduler.min_lr_ratio

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1 / warmup_steps, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _snapshot_run_inputs(
    config: ProjectConfig,
    store: ArtifactStore,
    *,
    paths: dict[str, Path],
    config_digest: str,
    data_digest_value: str,
    mapping: ClassMapping,
    resume: bool,
) -> None:
    if resume:
        return
    write_resolved_config(config, store.path("resolved_config.yaml"))
    for key, destination in (
        ("class_mapping", "data/class_to_idx.json"),
        ("train_manifest", "data/train_manifest.json"),
        ("val_manifest", "data/val_manifest.json"),
    ):
        shutil.copy2(paths[key], store.path(destination))
    atomic_write_bytes(
        store.path("data/manifest_digest.json"),
        (
            json.dumps(
                {
                    "config_digest": config_digest,
                    "data_digest": data_digest_value,
                    "class_mapping_digest": mapping.digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        overwrite=False,
    )
    atomic_write_bytes(
        store.path("environment/snapshot.json"),
        (json.dumps(collect_environment_snapshot(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        overwrite=False,
    )


def _preprocessing_spec(config: ProjectConfig) -> dict[str, object]:
    return {
        "image_size": config.data.image_size,
        "resize_short_side": config.data.eval_transform.resize_short_side,
        "center_crop": config.data.eval_transform.center_crop,
        "input_shape": [3, config.data.image_size, config.data.image_size],
        "dtype": "float32",
        "normalization": "openai_clip_official",
        "test_time_augmentation": False,
    }


def _resolve_device(raw: str) -> torch.device:
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AssemblyError(f"Configured CUDA device is unavailable: {raw}")
    if device.type not in {"cpu", "cuda"}:
        raise AssemblyError(f"Unsupported training device: {raw}")
    return device


def _reject_unintegrated_upper_bound_modules(config: ProjectConfig) -> None:
    unsupported: list[str] = []
    if config.noise.pseudolabel.enabled:
        unsupported.append("noise.pseudolabel")
    if config.noise.gradient_projection.enabled:
        unsupported.append("noise.gradient_projection")
    if config.noise.partition.method == "adaptive_mixture":
        unsupported.append("noise.partition.adaptive_mixture")
    if config.loss.logit_adjustment.enabled:
        unsupported.append("loss.logit_adjustment")
    if config.model.head.prototype_init.method == "multi_prototype":
        unsupported.append("model.head.prototype_init.multi_prototype")
    if unsupported:
        raise AssemblyError(
            "Upper-bound modules are intentionally not integrated into the B0-B6 runtime: "
            + ", ".join(unsupported)
        )


def _concrete_path(raw: str, field: str) -> Path:
    if raw.startswith("${oc.env:"):
        raise AssemblyError(f"{field} is unresolved; set its environment variable.")
    return Path(raw).expanduser().resolve()


def _optional_path(raw: str | None, fallback: Path) -> Path:
    return fallback.resolve() if raw is None else Path(raw).expanduser().resolve()
