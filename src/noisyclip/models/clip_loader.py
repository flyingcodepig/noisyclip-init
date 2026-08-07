"""Auditable lazy loading for the official OpenAI CLIP ViT-B/32 image model."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED = frozenset({"openai", "openai_clip_official"})
OPENAI_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
OPENAI_VIT_B32_URL = (
    f"https://openaipublic.azureedge.net/clip/models/{OPENAI_VIT_B32_SHA256}/ViT-B-32.pt"
)


class ClipBackend(Protocol):
    """Dependency-injection protocol for loading CLIP without network in tests."""

    package_version: str

    def load(
        self,
        name: str,
        *,
        device: str | torch.device,
        jit: bool,
        download_root: str | None,
    ) -> tuple[nn.Module, Callable[[Any], Any] | None]:
        """Load a CLIP model and preprocessing callable."""


@dataclass(frozen=True, slots=True)
class ClipWeightMetadata:
    """Auditable metadata for one official CLIP weight file.

    Attributes:
        model_name: Must be `ViT-B/32`.
        source: Official source identifier, either `openai` or
            `openai_clip_official`.
        download_identifier: Stable official download URL for ViT-B/32.
        file_path: Local resolved weight path, or `None` when a fake backend
            does not expose a file.
        sha256: SHA256 hex digest of the local file, or `None` for injected
            fake backends that intentionally avoid filesystem weights.
        package_version: Version string reported by the `clip` package or fake
            backend.
    """

    model_name: str
    source: str
    download_identifier: str
    file_path: str | None
    sha256: str | None
    package_version: str


@dataclass(frozen=True, slots=True)
class LoadedClipModel:
    """Loaded official CLIP model plus preprocessing and source metadata."""

    model: nn.Module
    preprocess: Callable[[Any], Any] | None
    metadata: ClipWeightMetadata


class OfficialClipBackend:
    """Lazy adapter around the official `clip` package."""

    def __init__(self) -> None:
        """Import the package only when real model loading is requested.

        Raises:
            RuntimeError: If the official OpenAI CLIP package is unavailable.
        """

        try:
            self._clip = importlib.import_module("clip")
        except ImportError as error:
            raise RuntimeError(
                "Official OpenAI CLIP package is not installed. Install it for real "
                "weight loading, or pass a fake ClipBackend in tests."
            ) from error
        self.package_version = str(getattr(self._clip, "__version__", "unknown"))

    def load(
        self,
        name: str,
        *,
        device: str | torch.device,
        jit: bool,
        download_root: str | None,
    ) -> tuple[nn.Module, Callable[[Any], Any] | None]:
        """Load `[ViT-B/32]` official weights through `clip.load`.

        Args:
            name: Must be `ViT-B/32`.
            device: Torch device for loading.
            jit: Passed through to the official package.
            download_root: Optional CLIP cache directory.

        Returns:
            `(model, preprocess)` from the official package.

        Raises:
            RuntimeError: If the package cannot load cached weights.
        """

        try:
            model, preprocess = self._clip.load(
                name,
                device=device,
                jit=jit,
                download_root=download_root,
            )
        except (RuntimeError, OSError, ValueError) as error:
            raise RuntimeError(
                "Failed to load official OpenAI CLIP ViT-B/32 weights. Ensure the "
                "official cache is present for offline runs and that network access is "
                "not required during tests."
            ) from error
        if not isinstance(model, nn.Module):
            raise TypeError("Official CLIP backend returned a non-module model.")
        return model, preprocess


def validate_clip_selection(model_name: str, pretrained: str) -> None:
    """Validate the only allowed backbone and official pretrained source.

    Args:
        model_name: Requested CLIP model name; only `ViT-B/32` is legal.
        pretrained: Requested source; must be `openai` or
            `openai_clip_official`.

    Raises:
        ValueError: If either field violates the competition boundary.
    """

    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(f"Only CLIP {ALLOWED_MODEL_NAME} is allowed, got {model_name!r}.")
    if pretrained not in ALLOWED_PRETRAINED:
        raise ValueError(
            f"Only official OpenAI CLIP weights are allowed; pretrained/source got {pretrained!r}."
        )


def load_clip_vit_b32(
    *,
    model_name: str = ALLOWED_MODEL_NAME,
    pretrained: str = "openai",
    device: str | torch.device = "cpu",
    cache_dir: Path | str | None = None,
    backend: ClipBackend | None = None,
    jit: bool = False,
    weight_path: Path | str | None = None,
) -> LoadedClipModel:
    """Load the official OpenAI CLIP ViT-B/32 model with auditable metadata.

    Args:
        model_name: Must be `ViT-B/32`.
        pretrained: Must identify official OpenAI weights.
        device: Torch device for the loaded module.
        cache_dir: Optional official CLIP cache root.
        backend: Optional injected backend for tests; fake backends must not
            download weights.
        jit: Passed to real backends.
        weight_path: Optional already-known local weight file to hash.

    Returns:
        `LoadedClipModel` containing the CLIP model, preprocessing callable, and
        metadata. The model may expose `encode_image(images) -> [B, D]`.

    Raises:
        ValueError: If model/source is not allowlisted.
        RuntimeError: If the official package or cached weights cannot be used.
        FileNotFoundError: If `weight_path` is provided but missing.
    """

    validate_clip_selection(model_name, pretrained)
    selected_backend = backend if backend is not None else OfficialClipBackend()
    explicit_weight_path = Path(weight_path).expanduser().resolve() if weight_path else None
    if explicit_weight_path is not None:
        cache_path = explicit_weight_path.parent
    elif cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
    else:
        cache_path = Path.home() / ".cache" / "clip"
    resolved_weight_path = explicit_weight_path
    if isinstance(selected_backend, OfficialClipBackend):
        resolved_weight_path = resolved_weight_path or cache_path / "ViT-B-32.pt"
        if not resolved_weight_path.is_file():
            raise RuntimeError(
                "Official OpenAI CLIP ViT-B/32 weights are not cached. Download the official "
                f"file during server bootstrap and place it at {resolved_weight_path}; runtime "
                "model loading is offline-only."
            )
    sha256 = _sha256_file(resolved_weight_path) if resolved_weight_path is not None else None
    if isinstance(selected_backend, OfficialClipBackend) and sha256 != OPENAI_VIT_B32_SHA256:
        raise RuntimeError(
            "Official OpenAI CLIP ViT-B/32 weight SHA256 mismatch: "
            f"expected {OPENAI_VIT_B32_SHA256}, got {sha256}. "
            "Replace the cached file with the verified official weight before retrying."
        )

    cache_root = str(cache_path)
    model, preprocess = selected_backend.load(
        model_name,
        device=device,
        jit=jit,
        download_root=cache_root,
    )
    metadata = ClipWeightMetadata(
        model_name=model_name,
        source=pretrained,
        download_identifier=OPENAI_VIT_B32_URL,
        file_path=str(resolved_weight_path) if resolved_weight_path is not None else None,
        sha256=sha256,
        package_version=selected_backend.package_version,
    )
    return LoadedClipModel(model=model, preprocess=preprocess, metadata=metadata)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"CLIP weight file not found for SHA256 audit: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
