"""Image encoder wrappers for official CLIP image features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from noisyclip.models.validation import require_image_batch
from noisyclip.utils.runtime_checks import value_checks_enabled


class CLIPImageBackbone(nn.Module):
    """Backbone wrapper that returns normalized CLIP image embeddings.

    Args:
        clip_model: Module exposing `encode_image(images) -> [B, D]`. Inputs
            must be `[B, 3, 224, 224]` floating-point tensors with finite values.
        embedding_dim: Optional expected output dimension `D`; when omitted,
            the wrapper tries `clip_model.visual.output_dim` or
            `clip_model.output_dim`.
        freeze: Whether all existing CLIP parameters should be frozen.

    Raises:
        AttributeError: If `clip_model` does not expose `encode_image`.
        ValueError: If `embedding_dim` cannot be inferred or output validation
            fails during `encode_image`.
    """

    embedding_dim: int

    def __init__(
        self,
        clip_model: nn.Module,
        *,
        embedding_dim: int | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not callable(getattr(clip_model, "encode_image", None)):
            raise AttributeError("clip_model must expose an encode_image(images) method.")
        self.clip_model = clip_model
        inferred_dim = (
            embedding_dim if embedding_dim is not None else _infer_embedding_dim(clip_model)
        )
        if inferred_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {inferred_dim}.")
        self.embedding_dim = inferred_dim
        if freeze:
            self.freeze()

    def encode_image(self, images: Tensor) -> Tensor:
        """Return `[B, D]` L2-normalized image features.

        Args:
            images: Floating-point tensor shaped `[B, 3, 224, 224]`, non-empty
                batch, finite values.

        Returns:
            Floating-point tensor shaped `[B, D]` with finite row-normalized
            embeddings.

        Raises:
            TypeError: If images are not floating-point.
            ValueError: If input shape, output shape, dtype, or finiteness is
                invalid.
        """

        require_image_batch(images)
        raw = cast(Any, self.clip_model).encode_image(images)
        if not isinstance(raw, Tensor):
            raise TypeError("clip_model.encode_image must return a torch.Tensor.")
        if raw.ndim != 2 or raw.shape[0] != images.shape[0] or raw.shape[1] != self.embedding_dim:
            raise ValueError(
                "clip_model.encode_image must return shape "
                f"[B, {self.embedding_dim}], got {tuple(raw.shape)}."
            )
        if not raw.is_floating_point():
            raise TypeError("clip_model.encode_image must return floating-point embeddings.")
        if value_checks_enabled() and not torch.isfinite(raw).all():
            raise ValueError("clip_model.encode_image returned NaN or Inf embeddings.")
        embedding = F.normalize(raw.float(), dim=1)
        if value_checks_enabled() and not torch.isfinite(embedding).all():
            raise ValueError("Normalized image embeddings contain NaN or Inf values.")
        return embedding

    def forward(self, images: Tensor) -> Tensor:
        """Return `[B, D]` normalized embeddings for `[B, 3, 224, 224]` images."""

        return self.encode_image(images)

    def freeze(self) -> None:
        """Set all current backbone parameters to `requires_grad=False`."""

        for parameter in self.parameters():
            parameter.requires_grad = False

    def requires_grad_report(self) -> dict[str, bool]:
        """Report actual `requires_grad` flags for every named parameter.

        Returns:
            Mapping from parameter name to whether it is trainable.
        """

        return {name: parameter.requires_grad for name, parameter in self.named_parameters()}

    def all_frozen(self) -> bool:
        """Return whether every current backbone parameter is frozen."""

        return all(not parameter.requires_grad for parameter in self.parameters())


def assert_backbone_trainability(
    backbone: nn.Module,
    *,
    allowed_trainable_prefixes: tuple[str, ...] = (),
) -> None:
    """Fail if unauthorized backbone parameters are trainable.

    Args:
        backbone: Backbone module to inspect.
        allowed_trainable_prefixes: Trainable parameter-name prefixes that are
            explicitly allowed, for example LoRA adapter parameters.

    Raises:
        ValueError: If any backbone parameter outside the allowlist is trainable.
    """

    unexpected = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad and not name.startswith(allowed_trainable_prefixes)
    ]
    if unexpected:
        raise ValueError(f"Unauthorized trainable backbone parameters: {unexpected}.")


def trainability_summary(module: nn.Module) -> Mapping[str, int]:
    """Return total and trainable parameter counts for a module."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}


def _infer_embedding_dim(module: nn.Module) -> int:
    visual = getattr(module, "visual", None)
    candidates = (
        getattr(visual, "output_dim", None),
        getattr(module, "output_dim", None),
        getattr(module, "embedding_dim", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
    raise ValueError("Could not infer CLIP image embedding dimension.")
