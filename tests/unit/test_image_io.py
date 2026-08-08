"""Tests for safe image IO and transforms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from noisyclip.config.schema import DataConfig
from noisyclip.data.image_io import ImageAuditError, inspect_image
from noisyclip.data.records import Batch
from noisyclip.data.transforms import build_transform, normalize_clip_tensor
from noisyclip.engine.device import move_batch_to_device


def _save_image(path: Path, mode: str, color: int | tuple[int, ...]) -> None:
    image = Image.new(mode, (40, 30), color)
    image.save(path)


def test_rgb_grayscale_and_rgba_images_are_readable(tmp_path: Path) -> None:
    """Pillow audit handles RGB, grayscale, and RGBA images."""

    cases = [
        ("rgb.png", "RGB", (10, 20, 30)),
        ("gray.png", "L", 127),
        ("rgba.png", "RGBA", (10, 20, 30, 128)),
    ]
    for filename, mode, color in cases:
        path = tmp_path / filename
        _save_image(path, mode, color)
        info = inspect_image(
            path,
            relative_path=filename,
            hash_file=True,
            allow_truncated_images=True,
            unreadable_policy="fail_audit",
        )
        assert info.readable
        assert info.width == 40
        assert info.height == 30
        assert info.file_sha256 is not None
        tensor = build_transform(DataConfig(expected_num_classes=3), split="val")(
            Image.open(path),
            sample_id=filename,
        )
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_truncated_image_policy_can_fail_or_skip(tmp_path: Path) -> None:
    """Truncated files fail in strict mode and produce a record in skip mode."""

    full = tmp_path / "full.jpg"
    _save_image(full, "RGB", (255, 0, 0))
    truncated = tmp_path / "truncated.jpg"
    payload = full.read_bytes()
    truncated.write_bytes(payload[: max(1, len(payload) // 3)])

    with pytest.raises(ImageAuditError, match="Unreadable image"):
        inspect_image(
            truncated,
            relative_path="truncated.jpg",
            hash_file=True,
            allow_truncated_images=False,
            unreadable_policy="fail_audit",
        )

    info = inspect_image(
        truncated,
        relative_path="truncated.jpg",
        hash_file=True,
        allow_truncated_images=False,
        unreadable_policy="skip_with_record",
    )
    assert not info.readable
    assert info.error is not None


def test_unreadable_image_is_reported(tmp_path: Path) -> None:
    """Non-image bytes produce explicit unreadable metadata."""

    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")

    info = inspect_image(
        bad,
        relative_path="bad.png",
        hash_file=False,
        allow_truncated_images=True,
        unreadable_policy="skip_with_record",
    )

    assert not info.readable
    assert info.width is None
    assert "Unreadable image" in str(info.error)


def test_uint8_transform_is_normalized_once_per_batch() -> None:
    """The optimized path preserves exact CLIP normalization on CPU fallback."""

    image = Image.new("RGB", (256, 256), (10, 20, 30))
    transform = build_transform(DataConfig(expected_num_classes=3), split="val", normalize=False)
    uint8_image = transform(image, sample_id="sample")
    assert uint8_image.dtype == torch.uint8
    batch = Batch(
        sample_ids=["sample"],
        image_weak=uint8_image.unsqueeze(0),
        image_strong=None,
        targets=torch.tensor([0]),
        class_ids=["0000"],
    )

    moved = move_batch_to_device(batch, "cpu")

    assert moved.image_weak is not None
    assert torch.equal(moved.image_weak[0], normalize_clip_tensor(uint8_image))
