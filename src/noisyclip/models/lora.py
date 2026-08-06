"""Controlled LoRA injection and merging for CLIP visual attention blocks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

VALID_PROJECTIONS = frozenset({"q", "k", "v"})
PROJECTION_TO_OFFSET = {"q": 0, "k": 1, "v": 2}


@dataclass(frozen=True, slots=True)
class LoraInjectionConfig:
    """Configuration for visual-transformer LoRA injection.

    Args:
        target_blocks: Zero-based visual transformer block indices. `None`
            means the last four blocks, or all blocks when fewer than four
            exist.
        target_projections: Subset of `q`, `k`, and `v`; default is `q` and
            `v`.
        rank: LoRA rank; must be positive when injection is enabled.
        alpha: LoRA alpha scaling; must be positive when injection is enabled.
        dropout: Dropout probability in `[0, 1)` applied to LoRA inputs.
    """

    target_blocks: tuple[int, ...] | None = None
    target_projections: tuple[str, ...] = ("q", "v")
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class LoraTrainableReport:
    """Trainable LoRA parameter accounting."""

    parameter_names: tuple[str, ...]
    parameter_count: int
    adapter_count: int


class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for `nn.MultiheadAttention` with fused `in_proj_weight`.

    The wrapped attention must use a fused `in_proj_weight` shaped
    `[3 * embed_dim, embed_dim]`. LoRA deltas are added only to configured
    q/k/v slices during forward, and `merge()` permanently folds them into the
    base weight for single-model export.

    Raises:
        TypeError: If the wrapped module is not `nn.MultiheadAttention`.
        ValueError: If projections, rank, alpha, dropout, or fused weights are
            invalid.
    """

    def __init__(
        self,
        base_attention: nn.MultiheadAttention,
        *,
        target_projections: Iterable[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base_attention, nn.MultiheadAttention):
            raise TypeError("LoRA injection only supports torch.nn.MultiheadAttention.")
        if base_attention.in_proj_weight is None:
            raise ValueError("LoRA requires fused in_proj_weight attention.")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        projections = tuple(target_projections)
        invalid = sorted(set(projections) - VALID_PROJECTIONS)
        if invalid:
            raise ValueError(f"Invalid LoRA projections: {invalid}.")
        if len(set(projections)) != len(projections):
            raise ValueError(f"LoRA projections must be unique, got {projections}.")

        self.base_attention = base_attention
        self.target_projections = projections
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.merged = False

        embed_dim = int(base_attention.embed_dim)
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        for projection in projections:
            a = nn.Parameter(torch.empty(rank, embed_dim))
            b = nn.Parameter(torch.zeros(embed_dim, rank))
            nn.init.kaiming_uniform_(a, a=5**0.5)
            self.lora_a[projection] = a
            self.lora_b[projection] = b

        for parameter in self.base_attention.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run attention with LoRA-adjusted fused q/k/v projection weights.

        Args:
            query: `[L, B, D]` or `[B, L, D]` attention query.
            key: Key tensor matching PyTorch `MultiheadAttention` semantics.
            value: Value tensor matching PyTorch `MultiheadAttention` semantics.
            key_padding_mask: Optional mask.
            need_weights: Whether attention weights should be returned.
            attn_mask: Optional attention mask.
            average_attn_weights: PyTorch attention weight averaging flag.
            is_causal: PyTorch causal attention hint.

        Returns:
            Tuple of attention output and optional weights with the same shapes
            as `nn.MultiheadAttention`.
        """

        attention = self.base_attention
        is_batched = query.dim() == 3
        if attention.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if self.merged:
            in_proj_weight = attention.in_proj_weight
        else:
            in_proj_weight = attention.in_proj_weight + self._delta_weight().to(
                dtype=attention.in_proj_weight.dtype,
                device=attention.in_proj_weight.device,
            )
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            attention.embed_dim,
            attention.num_heads,
            in_proj_weight,
            attention.in_proj_bias,
            attention.bias_k,
            attention.bias_v,
            attention.add_zero_attn,
            attention.dropout if self.training else 0.0,
            attention.out_proj.weight,
            attention.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if attention.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights

    def merge(self) -> None:
        """Permanently add LoRA deltas into fused `in_proj_weight`.

        Raises:
            RuntimeError: If called twice on the same adapter.
        """

        if self.merged:
            raise RuntimeError("LoRA adapter has already been merged.")
        with torch.no_grad():
            self.base_attention.in_proj_weight.add_(
                self._delta_weight().to(
                    dtype=self.base_attention.in_proj_weight.dtype,
                    device=self.base_attention.in_proj_weight.device,
                )
            )
        self.merged = True
        for parameter in self.lora_a.parameters():
            parameter.requires_grad = False
        for parameter in self.lora_b.parameters():
            parameter.requires_grad = False

    def _delta_weight(self) -> Tensor:
        embed_dim = int(self.base_attention.embed_dim)
        delta = torch.zeros(
            3 * embed_dim,
            embed_dim,
            dtype=self.lora_a[self.target_projections[0]].dtype,
            device=self.lora_a[self.target_projections[0]].device,
        )
        for projection in self.target_projections:
            start = PROJECTION_TO_OFFSET[projection] * embed_dim
            stop = start + embed_dim
            update = self.lora_b[projection] @ self.lora_a[projection]
            delta[start:stop, :] = update * self.scaling
        return delta


def inject_lora_into_visual_transformer(
    model: nn.Module,
    config: LoraInjectionConfig | None = None,
) -> LoraTrainableReport:
    """Inject LoRA into configured CLIP visual transformer attention blocks.

    Args:
        model: CLIP model or visual module exposing
            `visual.transformer.resblocks` or `transformer.resblocks`.
        config: Injection policy. Defaults to last four visual blocks, q/v,
            rank 8, alpha 8.

    Returns:
        Names and counts for trainable LoRA parameters.

    Raises:
        AttributeError: If visual transformer blocks cannot be found.
        IndexError: If a configured block index is out of range.
        ValueError: If projections or trainability are invalid.
    """

    policy = config or LoraInjectionConfig()
    _validate_lora_policy(policy)
    blocks = _get_visual_resblocks(model)
    block_indices = _resolve_target_blocks(len(blocks), policy.target_blocks)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block_index in block_indices:
        block = blocks[block_index]
        attention = getattr(block, "attn", None)
        if isinstance(attention, LoRAMultiheadAttention):
            raise ValueError(f"Block {block_index} already has a LoRA adapter.")
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError(f"Block {block_index} attention must be nn.MultiheadAttention.")
        block.attn = LoRAMultiheadAttention(
            attention,
            target_projections=policy.target_projections,
            rank=policy.rank,
            alpha=policy.alpha,
            dropout=policy.dropout,
        )

    report = lora_trainable_report(model)
    assert_only_lora_trainable(model)
    return report


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge all LoRA adapters and replace wrappers with plain attention.

    Args:
        model: Module containing injected visual LoRA adapters.

    Returns:
        Number of adapters merged. Calling this after all adapters have already
        been replaced is idempotent and returns `0`.
    """

    blocks = _get_visual_resblocks(model)
    merged = 0
    for block in blocks:
        attention = getattr(block, "attn", None)
        if isinstance(attention, LoRAMultiheadAttention):
            attention.merge()
            block.attn = attention.base_attention
            merged += 1
    return merged


def lora_trainable_report(model: nn.Module) -> LoraTrainableReport:
    """Return trainable LoRA parameter names and counts.

    Args:
        model: Module to inspect.

    Returns:
        Report containing only parameters whose names include `.lora_` and have
        `requires_grad=True`.
    """

    names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in f".{name}"
    )
    count = sum(parameter.numel() for name, parameter in model.named_parameters() if name in names)
    adapter_count = sum(
        1 for module in model.modules() if isinstance(module, LoRAMultiheadAttention)
    )
    return LoraTrainableReport(
        parameter_names=names, parameter_count=count, adapter_count=adapter_count
    )


def assert_only_lora_trainable(model: nn.Module) -> None:
    """Fail if any trainable model parameter is not a LoRA parameter."""

    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" not in f".{name}"
    ]
    if unexpected:
        raise ValueError(f"Unauthorized trainable backbone parameters: {unexpected}.")


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return all currently trainable parameter names in deterministic order."""

    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def _get_visual_resblocks(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    candidate = getattr(model, "clip_model", model)
    visual = getattr(candidate, "visual", candidate)
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if not isinstance(resblocks, (nn.ModuleList, list)):
        raise AttributeError("Model must expose visual.transformer.resblocks.")
    return resblocks


def _resolve_target_blocks(
    num_blocks: int, target_blocks: tuple[int, ...] | None
) -> tuple[int, ...]:
    if num_blocks <= 0:
        raise ValueError("Visual transformer must contain at least one block.")
    if target_blocks is None:
        start = max(0, num_blocks - 4)
        return tuple(range(start, num_blocks))
    if not target_blocks:
        raise ValueError("target_blocks cannot be empty when LoRA is enabled.")
    if len(set(target_blocks)) != len(target_blocks):
        raise ValueError(f"target_blocks must be unique, got {target_blocks}.")
    for index in target_blocks:
        if index < 0 or index >= num_blocks:
            raise IndexError(f"target block {index} is out of range for {num_blocks} blocks.")
    return target_blocks


def _validate_lora_policy(policy: LoraInjectionConfig) -> None:
    if policy.rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {policy.rank}.")
    if policy.alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {policy.alpha}.")
    if not 0.0 <= policy.dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {policy.dropout}.")
    invalid = sorted(set(policy.target_projections) - VALID_PROJECTIONS)
    if invalid:
        raise ValueError(f"Invalid LoRA projections: {invalid}.")
    if len(set(policy.target_projections)) != len(policy.target_projections):
        raise ValueError(f"LoRA projections must be unique, got {policy.target_projections}.")


def has_lora_adapters(model: nn.Module) -> bool:
    """Return whether a model currently contains unmerged LoRA adapters."""

    return any(isinstance(module, LoRAMultiheadAttention) for module in model.modules())


def adapter_debug_state(model: nn.Module) -> dict[str, Any]:
    """Return lightweight adapter state for tests and audit logs."""

    return {
        "has_lora": has_lora_adapters(model),
        "trainable_names": trainable_parameter_names(model),
        "lora_trainable_count": lora_trainable_report(model).parameter_count,
    }
