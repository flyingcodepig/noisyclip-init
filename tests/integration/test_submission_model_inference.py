"""End-to-end synthetic test for exported-model test inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from noisyclip.data.catalog import catalog_from_mapping, write_class_mapping
from noisyclip.data.manifests import make_sample_id, write_manifest
from noisyclip.data.records import SampleRecord
from noisyclip.models.backbone import CLIPImageBackbone
from noisyclip.models.classifier import LinearClassifierHead
from noisyclip.models.export import export_student_model
from noisyclip.models.student import NoisyCLIPStudent
from noisyclip.submission.inference import run_packaged_submission_inference


class TinyClip(nn.Module):
    """Small injected CLIP-shaped encoder used without network or weights."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.output_dim = 4
        self.visual.projection = nn.Linear(3, 4)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized `[B,4]` features from image channel means."""

        means = images.mean(dim=(2, 3))
        return F.normalize(self.visual.projection(means), dim=1)


class FakeClipBackend:
    """Fresh TinyClip factory satisfying the official-loader test protocol."""

    package_version = "test-only"

    def load(
        self,
        name: str,
        *,
        device: str | torch.device,
        jit: bool,
        download_root: str | None,
    ) -> tuple[nn.Module, Any]:
        """Return a local fake architecture; exported state supplies its weights."""

        del name, jit, download_root
        return TinyClip().to(device), None


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def test_exported_student_runs_real_image_forward_and_writes_csv(tmp_path: Path) -> None:
    """The formal path predicts from images rather than embedded prediction lists."""

    torch.manual_seed(31)
    mapping_dict = {"0001": 0, "0002": 1}
    catalog = catalog_from_mapping(mapping_dict)
    mapping_path = write_class_mapping(catalog, tmp_path / "class_to_idx.json")

    test_root = tmp_path / "test"
    test_root.mkdir()
    _save_image(test_root / "a.jpg", (20, 30, 40))
    _save_image(test_root / "b.jpg", (210, 180, 90))
    records = [
        SampleRecord(
            sample_id=make_sample_id(filename),
            relative_path=filename,
            split="test",
            class_id=None,
            target=None,
            file_sha256=None,
            width=32,
            height=32,
            readable=True,
        )
        for filename in ("a.jpg", "b.jpg")
    ]
    manifest_path = write_manifest(records, tmp_path / "test_manifest.json")

    student = NoisyCLIPStudent(
        backbone=CLIPImageBackbone(TinyClip(), freeze=True),
        head=LinearClassifierHead(embedding_dim=4, num_classes=2),
        stage="B0",
    )
    model_path = export_student_model(
        student,
        tmp_path / "model.pt",
        class_to_idx=mapping_dict,
        mapping_digest=catalog.digest,
        clip_weight_metadata={
            "model_name": "ViT-B/32",
            "source": "openai",
            "sha256": "0" * 64,
            "package_version": "test-only",
        },
    )

    report = run_packaged_submission_inference(
        model_path,
        tmp_path / "predictions",
        test_manifest_path=manifest_path,
        test_root=test_root,
        class_mapping_path=mapping_path,
        device="cpu",
        batch_size=2,
        num_workers=0,
        backend=FakeClipBackend(),
    )

    csv_path = tmp_path / "predictions" / "pred_results.csv"
    rows = [line.split(",") for line in csv_path.read_text(encoding="utf-8").splitlines()]
    assert report.valid
    assert [row[0] for row in rows] == ["a.jpg", "b.jpg"]
    assert all(row[1] in mapping_dict for row in rows)
