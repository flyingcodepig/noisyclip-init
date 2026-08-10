"""Two-batch synthetic training integration tests for Agent E."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from noisyclip.config.loader import load_config_from_mapping
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.engine.context import RunContext
from noisyclip.engine.trainer import Trainer, TrainerComponents
from noisyclip.losses.outputs import LossOutput
from noisyclip.models.classifier import LinearClassifierHead
from noisyclip.models.export import export_student_model
from noisyclip.models.student import NoisyCLIPStudent
from noisyclip.noise.state import JsonSampleStateStore, SampleState
from noisyclip.noise.trust import ClasswiseTrustAggregator
from noisyclip.submission.package import load_exported_model_package
from noisyclip.tracking.artifacts import ArtifactStore
from noisyclip.tracking.manifest import RunManifest
from noisyclip.utils.hashing import stable_hash

CLASS_TO_IDX = {"0001": 0, "0002": 1, "0003": 2}
CLIP_METADATA = {
    "model_name": "ViT-B/32",
    "source": "openai",
    "download_identifier": "official-test-fixture",
    "file_path": "fixture/ViT-B-32.pt",
    "sha256": "a" * 64,
    "package_version": "fake",
}


def tiny_config(
    *, epochs: int = 1, noise_enabled: bool = False, warmup_epochs: int = 3
) -> object:
    """Return a minimal F02 config for CPU/fp32 synthetic training."""

    return load_config_from_mapping(
        {
            "experiment": {},
            "paths": {"run_root": "runs"},
            "data": {"expected_num_classes": 3},
            "model": {},
            "noise": {
                "enabled": noise_enabled,
                "warmup_epochs": warmup_epochs,
                "signals": {"ema_loss": {"enabled": noise_enabled, "coefficient": 1.0}},
                "partition": {"min_samples_per_class": 2},
            },
            "loss": {},
            "trainer": {
                "epochs": epochs,
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 3,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
            },
            "evaluation": {},
            "tracking": {"minimum_free_disk_gib": 0.000000001, "tensorboard": False},
            "submission": {},
        }
    )


def tiny_records(*, split: str = "train") -> list[SampleRecord]:
    """Return six stable synthetic records across three classes."""

    records: list[SampleRecord] = []
    for index in range(6):
        class_index = index % 3
        class_id = f"{class_index + 1:04d}"
        records.append(
            SampleRecord(
                sample_id=f"s{index}",
                relative_path=f"{class_id}/s{index}.png",
                split=split,
                class_id=None if split == "test" else class_id,
                target=None if split == "test" else class_index,
                file_sha256=None,
                width=224,
                height=224,
                readable=True,
            )
        )
    return records


def tiny_batches(records: list[SampleRecord]) -> list[Batch]:
    """Build two deterministic training batches from synthetic records."""

    batches: list[Batch] = []
    for offset in (0, 3):
        chunk = records[offset : offset + 3]
        image = torch.zeros((3, 3, 224, 224), dtype=torch.float32)
        for row, record in enumerate(chunk):
            target = int(record.target or 0)
            image[row, :, 0, 0] = torch.tensor(
                [1.0 + target, 0.5 + row, 0.25 + offset],
                dtype=torch.float32,
            )
        batches.append(
            Batch(
                sample_ids=[record.sample_id for record in chunk],
                image_weak=image,
                image_strong=None,
                targets=torch.tensor([int(record.target or 0) for record in chunk]),
                class_ids=[str(record.class_id) for record in chunk],
            )
        )
    return batches


def tiny_components(
    tmp_path: Path,
    *,
    epochs: int = 1,
    noise_enabled: bool = False,
    warmup_epochs: int = 3,
) -> tuple[object, TrainerComponents]:
    """Create config and injected trainer components for synthetic tests."""

    torch.manual_seed(1)
    config = tiny_config(
        epochs=epochs,
        noise_enabled=noise_enabled,
        warmup_epochs=warmup_epochs,
    )
    model = ExportableTinyStudent()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    records = tiny_records()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_context = RunContext(
        run_id="run",
        run_dir=run_dir,
        seed=1,
        num_classes=3,
        class_to_idx=CLASS_TO_IDX,
        config_digest=stable_hash({"config": "tiny"}),
        data_digest=stable_hash({"data": "tiny"}),
    )
    components = TrainerComponents(
        model=model,
        optimizer=optimizer,
        loss=TinyLoss(),
        train_loader=tiny_batches(records),
        val_loader=tiny_batches(records),
        train_records=records,
        state_store=JsonSampleStateStore(run_dir / "sample_state", [r.sample_id for r in records]),
        run_context=run_context,
        artifact_store=ArtifactStore(run_dir),
        run_manifest=RunManifest(run_dir, {}),
    )
    return config, components


def test_two_batch_train_updates_head_freezes_backbone_and_exports_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Two synthetic batches update the head, keep backbone frozen, and export one model."""

    config, components = tiny_components(tmp_path)
    before_head = components.model.head.linear.weight.detach().clone()
    before_backbone = components.model.backbone.proj.weight.detach().clone()
    result = Trainer(config=config, components=components, device="cpu").fit()
    assert result.global_step == 2
    assert result.exported_model == tmp_path / "run" / "artifacts" / "model.pt"
    assert (tmp_path / "run" / "checkpoints" / "best_top1.pt").is_file()
    assert (tmp_path / "run" / "checkpoints" / "epoch_0000.pt").is_file()
    checkpoint = torch.load(result.last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["sample_state_epoch"] is None
    assert not (tmp_path / "run" / "sample_state" / "manifest.json").exists()
    assert not list((tmp_path / "run" / "sample_state").glob("epoch_*.json"))
    assert not torch.equal(components.model.head.linear.weight.detach(), before_head)
    assert torch.equal(components.model.backbone.proj.weight.detach(), before_backbone)
    package = load_exported_model_package(result.exported_model)
    assert package.num_classes == 3
    assert not (tmp_path / "run" / "artifacts" / "teacher.pt").exists()


def test_noise_enabled_training_still_persists_prediction_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Warmup persists compact history without arbitrary trust weighting."""

    config, components = tiny_components(tmp_path, noise_enabled=True)
    result = Trainer(config=config, components=components, device="cpu").fit()

    checkpoint = torch.load(result.last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["sample_state_epoch"] == 0
    states = components.state_store.load_all()
    assert len(states) == len(components.train_records)
    assert all(state.ema_probs is None for state in states)
    assert all(len(state.prediction_history) == 1 for state in states)
    assert {state.partition for state in states} == {"trusted"}
    assert {state.supervised_weight for state in states} == {1.0}


def test_noise_enabled_training_updates_trust_when_warmup_is_due(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The first due trust update collects signals while keeping state compact."""

    config, components = tiny_components(tmp_path, noise_enabled=True, warmup_epochs=0)
    components.trust_aggregator = ClasswiseTrustAggregator.from_config(config.noise)
    Trainer(config=config, components=components, device="cpu").fit()

    states = components.state_store.load_all()
    assert all(state.ema_loss > 0.0 for state in states)
    assert all(state.seen_count == 1 for state in states)
    assert all(state.ema_probs is None for state in states)


class TinyLoss:
    """Finite weighted CE loss with detached per-sample supervised losses."""

    def __call__(
        self,
        batch: Batch,
        student_weak: object,
        student_strong: object | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> LossOutput:
        """Return scalar CE and per-sample CE for `[B, C]` logits."""

        del student_strong, teacher_embedding, sample_states, epoch
        logits = student_weak.logits
        if batch.targets is None:
            raise ValueError("targets required")
        per_sample = F.cross_entropy(logits, batch.targets, reduction="none")
        total = per_sample.mean()
        return LossOutput(
            total=total,
            components={"loss/ce": total},
            per_sample_supervised=per_sample.detach(),
        )


class TinyClipLike(nn.Module):
    """Small CLIP-like encoder exposing `encode_image`."""

    embedding_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, self.embedding_dim)

    def encode_image(self, images: Tensor) -> Tensor:
        """Return L2-normalized `[B, 4]` features from first-pixel channels."""

        return F.normalize(self.proj(images[:, :, 0, 0].float()), dim=1)


class ExportableTinyStudent(NoisyCLIPStudent):
    """NoisyCLIP student with test-only export metadata injection."""

    def __init__(self) -> None:
        super().__init__(
            backbone=TinyClipLike(),
            head=LinearClassifierHead(embedding_dim=4, num_classes=3),
            stage="B0",
        )

    def export_single_model(self, destination: Path) -> Path:
        """Export one formal single-model artifact with fake official metadata."""

        return export_student_model(
            self,
            destination,
            class_to_idx=CLASS_TO_IDX,
            clip_weight_metadata=CLIP_METADATA,
        )
