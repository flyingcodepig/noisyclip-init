"""End-to-end assembly coverage for the executable B0/B1/B2 path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from noisyclip.config.loader import load_config_from_mapping, write_resolved_config
from noisyclip.config.schema import ProjectConfig
from noisyclip.data.catalog import build_class_catalog, write_class_mapping
from noisyclip.data.manifests import make_sample_id, write_manifest
from noisyclip.data.records import SampleRecord
from noisyclip.engine.assembly import (
    assemble_trainer,
    build_feature_cache_from_config,
    evaluate_checkpoint,
    run_training,
)
from noisyclip.models.export import load_export_package


class TinyBlock(nn.Module):
    """Minimal attention block accepted by the LoRA injector."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(4, num_heads=2)


class TinyOfficialClip(nn.Module):
    """Small differentiable CLIP-shaped image encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.output_dim = 4
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList([TinyBlock() for _ in range(4)])
        self.projection = nn.Linear(3, 4, bias=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Project global RGB means to four dimensions."""

        return self.projection(images.mean(dim=(2, 3)))


class FakeBackend:
    """Offline backend used to verify assembly without downloading weights."""

    package_version = "fake"

    def load(
        self,
        name: str,
        *,
        device: str | torch.device,
        jit: bool,
        download_root: str | None,
    ) -> tuple[nn.Module, Any]:
        """Return a fresh fake model for every requested assembly."""

        del name, jit, download_root
        return TinyOfficialClip().to(device), None


def _fixture_config(
    tmp_path: Path, *, stage: str, feature_cache: bool = False
) -> tuple[Path, ProjectConfig]:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    run_root = tmp_path / "runs"
    cache_root = tmp_path / "cache"
    for path in (train_root, test_root, run_root / "data", cache_root):
        path.mkdir(parents=True, exist_ok=True)
    catalog = build_class_catalog(["0000", "0001", "0002"], expected_num_classes=3)
    write_class_mapping(catalog, run_root / "data" / "class_to_idx.json")
    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    for target, class_id in enumerate(catalog.class_to_idx):
        class_dir = train_root / class_id
        class_dir.mkdir()
        for item in range(3):
            relative = f"{class_id}/{item}.png"
            Image.new("RGB", (224, 224), color=(30 + target * 50, 40 + item, 80)).save(
                train_root / relative
            )
            split = "val" if item == 2 else "train"
            record = SampleRecord(
                sample_id=make_sample_id(relative),
                relative_path=relative,
                split=split,
                class_id=class_id,
                target=target,
                file_sha256=None,
                width=224,
                height=224,
                readable=True,
            )
            (val_records if split == "val" else train_records).append(record)
    write_manifest(train_records, run_root / "data" / "train_manifest.json")
    write_manifest(val_records, run_root / "data" / "val_manifest.json")
    model: dict[str, Any] = {}
    if stage in {"B1", "B2"}:
        model["head"] = {
            "type": "cosine",
            "temperature_init": 10.0,
            "temperature_min": 1.0,
            "temperature_max": 100.0,
            "prototype_init": {"enabled": True, "method": "trimmed_mean"},
        }
    if stage == "B2":
        model["lora"] = {
            "enabled": True,
            "target_blocks": [-2, -1],
            "target_projections": ["q", "v"],
            "rank": 2,
            "alpha": 4,
        }
    config = load_config_from_mapping(
        {
            "experiment": {"name": stage.lower(), "seed": 7},
            "paths": {
                "train_root": str(train_root),
                "test_root": str(test_root),
                "run_root": str(run_root),
                "cache_root": str(cache_root),
            },
            "data": {"expected_num_classes": 3},
            "model": model,
            "noise": {"partition": {"min_samples_per_class": 2}},
            "loss": {},
            "trainer": {
                "epochs": 1,
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 3,
                "num_workers": 0,
                "early_stopping": {"enabled": False},
                "frozen_feature_cache": {
                    "enabled": feature_cache,
                    "directory": str(tmp_path / "features") if feature_cache else None,
                },
            },
            "evaluation": {},
            "tracking": {
                "minimum_free_disk_gib": 0.000000001,
                "tensorboard": False,
            },
            "submission": {},
        }
    )
    config_path = tmp_path / f"{stage.lower()}.yaml"
    write_resolved_config(config, config_path)
    return config_path, config


def test_b0_real_assembly_trains_exports_and_writes_done(tmp_path: Path) -> None:
    """The library entry point runs from fixed manifests through formal export."""

    config_path, _ = _fixture_config(tmp_path, stage="B0")
    result = run_training(config_path, run_id="b0-run", clip_backend=FakeBackend())

    assert result.epochs_completed == 1
    assert result.last_checkpoint.is_file()
    assert result.exported_model is not None
    package = load_export_package(result.exported_model)
    assert package["mapping_digest"]
    assert package["clip_weight_metadata"]["source"] == "openai"
    assert (tmp_path / "runs" / "b0-run" / "DONE").is_file()
    evaluated = evaluate_checkpoint(
        tmp_path / "runs" / "b0-run",
        result.last_checkpoint,
        clip_backend=FakeBackend(),
    )
    assert evaluated.metrics["val/top1"] is not None


def test_b0_b1_b2_configs_all_assemble_with_expected_trainability(tmp_path: Path) -> None:
    """All three required baseline configurations construct without GPU or network."""

    for stage in ("B0", "B1", "B2"):
        stage_root = tmp_path / stage.lower()
        _, config = _fixture_config(stage_root, stage=stage)
        trainer = assemble_trainer(
            config,
            run_id=f"{stage.lower()}-assemble",
            clip_backend=FakeBackend(),
        )
        model = trainer.components.model
        assert model.stage == stage
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        assert any(name.startswith("head.") for name in trainable)
        assert any(".lora_" in f".{name}" for name in trainable) is (stage == "B2")


def test_b0_and_b1_consume_provenance_bound_frozen_features(tmp_path: Path) -> None:
    """The cache builder feeds complete B0/B1 runs without image batches."""

    for stage in ("B0", "B1"):
        stage_root = tmp_path / stage.lower()
        config_path, _ = _fixture_config(stage_root, stage=stage, feature_cache=True)
        cache_root = build_feature_cache_from_config(config_path, clip_backend=FakeBackend())

        assert (cache_root / "metadata.json").is_file()
        assert (cache_root / "train_epoch_0000.pt").is_file()
        assert (cache_root / "train_eval.pt").is_file()
        assert (cache_root / "val.pt").is_file()

        result = run_training(
            config_path,
            run_id=f"{stage.lower()}-cached-run",
            clip_backend=FakeBackend(),
        )
        assert result.epochs_completed == 1
        assert result.global_step == 2
        evaluated = evaluate_checkpoint(
            stage_root / "runs" / f"{stage.lower()}-cached-run",
            result.last_checkpoint,
            clip_backend=FakeBackend(),
        )
        assert evaluated.metrics["val/top1"] is not None
