"""Tests for lazy official CLIP loading boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from noisyclip.models.clip_loader import OfficialClipBackend, load_clip_vit_b32


class TinyClip(nn.Module):
    """Tiny fake CLIP module for dependency-injected loader tests."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.output_dim = 4

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return fake `[B, 4]` embeddings from `[B,3,224,224]` images."""

        return torch.ones((images.shape[0], 4), dtype=images.dtype, device=images.device)


class FakeBackend:
    """Backend that never downloads and records load arguments."""

    package_version = "fake-clip"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def load(
        self,
        name: str,
        *,
        device: str | torch.device,
        jit: bool,
        download_root: str | None,
    ) -> tuple[nn.Module, Any]:
        """Return a fake CLIP model without using network or disk caches."""

        del device, jit
        self.calls.append((name, download_root))
        return TinyClip(), None


def test_loader_rejects_non_vit_b32_and_non_openai_source() -> None:
    """The competition backbone/source allowlist is enforced before backend use."""

    with pytest.raises(ValueError, match="ViT-B/32"):
        load_clip_vit_b32(model_name="RN50", backend=FakeBackend())

    with pytest.raises(ValueError, match="official OpenAI"):
        load_clip_vit_b32(pretrained="laion", backend=FakeBackend())


def test_loader_uses_injected_backend_and_hashes_weight_file(tmp_path: Path) -> None:
    """Injected backends avoid downloads while metadata remains auditable."""

    weight_path = tmp_path / "ViT-B-32.pt"
    weight_path.write_bytes(b"fake weight bytes")
    backend = FakeBackend()

    loaded = load_clip_vit_b32(cache_dir=tmp_path, backend=backend, weight_path=weight_path)

    assert isinstance(loaded.model, TinyClip)
    assert backend.calls == [("ViT-B/32", str(tmp_path))]
    assert loaded.metadata.model_name == "ViT-B/32"
    assert loaded.metadata.source == "openai"
    assert loaded.metadata.sha256 == hashlib.sha256(b"fake weight bytes").hexdigest()
    assert loaded.metadata.file_path == str(weight_path.resolve())


def test_missing_weight_path_fails_clearly(tmp_path: Path) -> None:
    """A requested local audit file must exist."""

    with pytest.raises(FileNotFoundError, match="SHA256 audit"):
        load_clip_vit_b32(backend=FakeBackend(), weight_path=tmp_path / "missing.pt")


def test_official_backend_import_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real backend reports a package installation problem at load time."""

    def fake_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)
    with pytest.raises(RuntimeError, match="Official OpenAI CLIP package"):
        OfficialClipBackend()
