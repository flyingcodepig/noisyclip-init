"""Exact, provenance-bound frozen CLIP feature caches for B0/B1."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from noisyclip.config.schema import ProjectConfig
from noisyclip.data.dataset import ManifestImageDataset, collate_batch
from noisyclip.data.image_io import build_image_loader, file_sha256
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.data.sampler import DeterministicSampler, build_sampler
from noisyclip.data.transforms import build_transform
from noisyclip.engine.device import BatchDeviceIterator
from noisyclip.utils.hashing import stable_hash


class FeatureCacheError(RuntimeError):
    """Raised when cache identity, contents, or completeness is unsafe."""


@dataclass(frozen=True, slots=True)
class ReferenceFeatureCache:
    """Validated deterministic train/validation references from a frozen cache."""

    root: Path
    signature: str
    train_eval: Tensor
    train_sample_ids: tuple[str, ...]
    train_targets: Tensor
    val_sample_ids: tuple[str, ...]
    val_by_sample: Mapping[str, Tensor]


def feature_cache_signature(
    config: ProjectConfig,
    *,
    data_digest: str,
    class_mapping_digest: str,
    clip_weight_sha256: str,
) -> str:
    """Hash only inputs that determine frozen embeddings, shared by B0 and B1."""

    return stable_hash(
        {
            "schema_version": 1,
            "data_digest": data_digest,
            "class_mapping_digest": class_mapping_digest,
            "clip_weight_sha256": clip_weight_sha256,
            "seed": config.experiment.seed,
            "epochs": config.trainer.epochs,
            "data": config.data.model_dump(mode="json"),
            "backbone": config.model.backbone.model_dump(mode="json"),
            "feature_dtype": "float32",
        }
    )


def build_frozen_feature_cache(
    destination: Path | str,
    *,
    config: ProjectConfig,
    model: Any,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    train_root: Path,
    device: torch.device,
    data_digest: str,
    class_mapping_digest: str,
    clip_weight_sha256: str,
) -> Path:
    """Build all epoch-specific train features plus deterministic eval features."""

    root = Path(destination).resolve()
    staging = root.with_name(f"{root.name}.building")
    if root.exists() or staging.exists():
        raise FileExistsError(f"Feature cache target or staging directory already exists: {root}")
    staging.mkdir(parents=True)
    ordered_train = sorted(train_records, key=lambda row: (row.relative_path, row.sample_id))
    ordered_val = sorted(val_records, key=lambda row: (row.relative_path, row.sample_id))
    image_loader = build_image_loader(config.data.decode_backend)
    train_dataset = ManifestImageDataset(
        ordered_train,
        data_root=train_root,
        split="train",
        image_weak_transform=build_transform(
            config.data,
            split="train",
            seed=config.experiment.seed,
            normalize=not config.data.normalize_on_device,
        ),
        training=True,
        image_loader=image_loader,
    )
    val_dataset = ManifestImageDataset(
        ordered_val,
        data_root=train_root,
        split="val",
        image_weak_transform=build_transform(
            config.data, split="val", normalize=not config.data.normalize_on_device
        ),
        image_loader=image_loader,
    )
    train_eval_dataset = ManifestImageDataset(
        ordered_train,
        data_root=train_root,
        split="train",
        image_weak_transform=build_transform(
            config.data, split="val", normalize=not config.data.normalize_on_device
        ),
        image_loader=image_loader,
    )
    train_loader = _image_loader(config, train_dataset)
    model.backbone.eval()
    files: dict[str, str] = {}
    for epoch in range(config.trainer.epochs):
        train_dataset.set_epoch(epoch)
        path = staging / f"train_epoch_{epoch:04d}.pt"
        _save_features(path, _encode_loader(model.backbone, train_loader, device))
        files[path.name] = file_sha256(path)
    for name, dataset in (("train_eval.pt", train_eval_dataset), ("val.pt", val_dataset)):
        path = staging / name
        _save_features(path, _encode_loader(model.backbone, _image_loader(config, dataset), device))
        files[name] = file_sha256(path)
    signature = feature_cache_signature(
        config,
        data_digest=data_digest,
        class_mapping_digest=class_mapping_digest,
        clip_weight_sha256=clip_weight_sha256,
    )
    metadata = {
        "schema_version": 1,
        "signature": signature,
        "feature_dtype": "float32",
        "embedding_dim": int(model.backbone.embedding_dim),
        "epochs": config.trainer.epochs,
        "train_sample_ids": [record.sample_id for record in ordered_train],
        "train_targets": [record.target for record in ordered_train],
        "train_class_ids": [record.class_id for record in ordered_train],
        "val_sample_ids": [record.sample_id for record in ordered_val],
        "val_targets": [record.target for record in ordered_val],
        "val_class_ids": [record.class_id for record in ordered_val],
        "files": files,
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(root)
    return root


class FrozenFeatureLoader:
    """Small iterable that memory-maps one exact feature tensor per epoch."""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str,
        batch_size: int,
        seed: int,
        expected_signature: str,
        verify_hashes: bool,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.batch_size = batch_size
        self.epoch = 0
        self.metadata = _load_metadata(self.root, expected_signature, verify_hashes)
        prefix = "train" if split == "train" else "val"
        self.sample_ids = [str(item) for item in self.metadata[f"{prefix}_sample_ids"]]
        self.targets = torch.tensor(self.metadata[f"{prefix}_targets"], dtype=torch.int64)
        self.class_ids = [str(item) for item in self.metadata[f"{prefix}_class_ids"]]
        records = [
            SampleRecord(
                sample_id=sid,
                relative_path=sid,
                split=split,
                class_id=cid,
                target=int(target),
                file_sha256=None,
                width=None,
                height=None,
                readable=True,
            )
            for sid, cid, target in zip(
                self.sample_ids, self.class_ids, self.targets.tolist(), strict=True
            )
        ]
        self.sampler: DeterministicSampler = build_sampler(
            records, shuffle=split == "train", seed=seed
        )
        self.dataset = self

    def __len__(self) -> int:
        return math.ceil(len(self.sample_ids) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[Batch]:
        filename = f"train_epoch_{self.epoch:04d}.pt" if self.split == "train" else "val.pt"
        features = load_feature_tensor(self.root / filename)
        if features.shape[0] != len(self.sample_ids):
            raise FeatureCacheError(f"Feature row count mismatch in {filename}.")
        indices = list(self.sampler)
        for offset in range(0, len(indices), self.batch_size):
            selected = indices[offset : offset + self.batch_size]
            index = torch.tensor(selected, dtype=torch.int64)
            yield Batch(
                sample_ids=[self.sample_ids[item] for item in selected],
                image_weak=None,
                image_strong=None,
                targets=self.targets[index],
                class_ids=[self.class_ids[item] for item in selected],
                embedding_weak=features[index],
            )


def load_feature_tensor(path: Path | str) -> Tensor:
    """Load a float32 rank-2 cache tensor using mmap when supported."""

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - older supported torch versions.
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, Tensor) or tensor.ndim != 2 or tensor.dtype != torch.float32:
        raise FeatureCacheError(f"Malformed feature tensor: {path}")
    if not torch.isfinite(tensor).all():
        raise FeatureCacheError(f"Feature tensor contains NaN or Inf: {path}")
    return tensor


def load_reference_feature_cache(
    root: Path | str,
    *,
    expected_signature: str,
    verify_hashes: bool,
) -> ReferenceFeatureCache:
    """Load only deterministic prototype and validation reference features."""

    cache_root = Path(root).resolve()
    metadata = _load_metadata(
        cache_root,
        expected_signature,
        verify_hashes,
        required_files={"train_eval.pt", "val.pt"},
    )
    train_eval = load_feature_tensor(cache_root / "train_eval.pt")
    val = load_feature_tensor(cache_root / "val.pt")
    train_targets = torch.tensor(metadata.get("train_targets"), dtype=torch.int64)
    train_ids = _sample_ids(metadata, "train_sample_ids")
    val_ids = _sample_ids(metadata, "val_sample_ids")
    if train_eval.shape[0] != len(train_ids) or train_targets.shape != (len(train_ids),):
        raise FeatureCacheError("Reference train_eval rows do not match cache metadata.")
    if val.shape[0] != len(val_ids):
        raise FeatureCacheError("Reference val rows do not match cache metadata.")
    return ReferenceFeatureCache(
        root=cache_root,
        signature=expected_signature,
        train_eval=train_eval,
        train_sample_ids=tuple(train_ids),
        train_targets=train_targets,
        val_sample_ids=tuple(val_ids),
        val_by_sample={sample_id: val[index] for index, sample_id in enumerate(val_ids)},
    )


def _image_loader(config: ProjectConfig, dataset: ManifestImageDataset) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
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


def _encode_loader(backbone: Any, loader: DataLoader[Any], device: torch.device) -> Tensor:
    parts: list[Tensor] = []
    with torch.inference_mode():
        for batch in BatchDeviceIterator(loader, device):
            if batch.image_weak is None:
                raise FeatureCacheError("Feature generation requires image tensors.")
            parts.append(backbone.encode_image(batch.image_weak).detach().cpu().float())
    if not parts:
        raise FeatureCacheError("Feature generation loader produced no batches.")
    return torch.cat(parts)


def _save_features(path: Path, features: Tensor) -> None:
    if features.ndim != 2 or features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise FeatureCacheError("Refusing to save malformed frozen features.")
    torch.save(features.contiguous(), path)


def _load_metadata(
    root: Path,
    expected_signature: str,
    verify_hashes: bool,
    *,
    required_files: set[str] | None = None,
) -> Mapping[str, Any]:
    path = root / "metadata.json"
    if not path.is_file():
        raise FeatureCacheError(f"Feature cache metadata is missing: {path}")
    metadata = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if metadata.get("schema_version") != 1 or metadata.get("signature") != expected_signature:
        raise FeatureCacheError("Feature cache identity does not match this run.")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise FeatureCacheError("Feature cache file manifest is malformed.")
    filenames = set(map(str, files)) if required_files is None else required_files
    missing_manifest_entries = filenames - set(map(str, files))
    if missing_manifest_entries:
        raise FeatureCacheError(
            f"Feature cache manifest lacks required files: {sorted(missing_manifest_entries)}"
        )
    for filename in sorted(filenames):
        expected_hash = files[filename]
        feature_path = root / str(filename)
        if not feature_path.is_file():
            raise FeatureCacheError(f"Feature cache file is missing: {feature_path}")
        if verify_hashes and file_sha256(feature_path) != expected_hash:
            raise FeatureCacheError(f"Feature cache hash mismatch: {feature_path}")
    return metadata


def _sample_ids(metadata: Mapping[str, Any], key: str) -> list[str]:
    raw = metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise FeatureCacheError(f"Feature cache {key} is malformed.")
    if len(set(raw)) != len(raw):
        raise FeatureCacheError(f"Feature cache {key} contains duplicates.")
    return raw
