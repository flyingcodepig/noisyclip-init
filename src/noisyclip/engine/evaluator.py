"""Validation evaluator producing metrics, per-class CSV, and confusion matrix."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.metrics.classification import (
    ClassificationMetrics,
    MetricValue,
    compute_classification_metrics,
)
from noisyclip.metrics.drift import feature_cosine_to_base
from noisyclip.metrics.robustness import augmentation_agreement, trusted_subset_top1


class EvaluatableModel(Protocol):
    """Protocol for validation models returning `ModelOutput` from batches."""

    def eval(self) -> EvaluatableModel:
        """Switch to evaluation mode and return self."""

    def __call__(self, images: Tensor) -> object:
        """Return an object exposing logits `[B, C]` and embedding `[B, D]`."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregated validation metrics and diagnostic tensors."""

    metrics: dict[str, float | None]
    metric_reasons: dict[str, str]
    per_class_accuracy: dict[int, MetricValue]
    confusion_matrix: Tensor


class Evaluator:
    """Evaluate one model over a fixed validation loader.

    Args:
        model: Callable model that returns logits and embeddings.
        num_classes: Positive number of validation classes.
        device: Torch device used for inference.

    Raises:
        ValueError: If `num_classes` is not positive.
    """

    def __init__(
        self,
        *,
        model: EvaluatableModel,
        num_classes: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(device)

    @torch.inference_mode()
    def evaluate(
        self,
        loader: Iterable[Batch],
        *,
        trusted_ids: set[str] | None = None,
        base_embeddings: MappingBySample | None = None,
    ) -> EvaluationResult:
        """Compute metrics from a fixed validation loader.

        Args:
            loader: Iterable yielding `Batch` objects with labels.
            trusted_ids: Optional sample IDs used for trusted-subset top-1.
            base_embeddings: Optional mapping from sample IDs to frozen CLIP
                embeddings `[D]` for drift.

        Returns:
            `EvaluationResult` with metric values in `[0, 1]` or `None`.

        Raises:
            ValueError: If a validation batch is unlabeled or tensors are
                malformed.
        """

        self.model.eval()
        logits_parts: list[Tensor] = []
        targets_parts: list[Tensor] = []
        strong_logits_parts: list[Tensor] = []
        trusted_masks: list[Tensor] = []
        embeddings: list[Tensor] = []
        base_parts: list[Tensor] = []
        saw_strong = True

        for batch in loader:
            if batch.targets is None:
                raise ValueError("Validation batch must include targets.")
            output = self.model(batch.image_weak.to(self.device))
            logits = _get_tensor_attr(output, "logits").detach().cpu()
            embedding = _get_tensor_attr(output, "embedding").detach().cpu()
            logits_parts.append(logits)
            targets_parts.append(batch.targets.detach().cpu())
            embeddings.append(embedding)

            if batch.image_strong is None:
                saw_strong = False
            else:
                strong_output = self.model(batch.image_strong.to(self.device))
                strong_logits_parts.append(_get_tensor_attr(strong_output, "logits").detach().cpu())

            if trusted_ids is not None:
                trusted_masks.append(
                    torch.tensor([sample_id in trusted_ids for sample_id in batch.sample_ids])
                )
            if base_embeddings is not None:
                for sample_id in batch.sample_ids:
                    base_parts.append(base_embeddings[sample_id].detach().cpu())

        if not logits_parts:
            raise ValueError("Validation loader produced no batches.")
        logits_all = torch.cat(logits_parts, dim=0)
        targets_all = torch.cat(targets_parts, dim=0)
        classification = compute_classification_metrics(
            logits_all,
            targets_all,
            num_classes=self.num_classes,
        )
        trusted_mask = torch.cat(trusted_masks, dim=0) if trusted_masks else None
        strong_logits = (
            torch.cat(strong_logits_parts, dim=0) if saw_strong and strong_logits_parts else None
        )
        base_tensor = torch.stack(base_parts, dim=0) if base_parts else None
        drift = feature_cosine_to_base(torch.cat(embeddings, dim=0), base_tensor)
        result_metrics = {
            "val/top1": classification.top1.value,
            "val/macro_accuracy": classification.macro_accuracy.value,
            "val/bottom_quartile_accuracy": classification.bottom_quartile_accuracy.value,
            "val/trusted_top1": trusted_subset_top1(logits_all, targets_all, trusted_mask).value,
            "val/augmentation_agreement": augmentation_agreement(logits_all, strong_logits).value,
            "val/feature_cosine_to_base": drift.value,
        }
        reasons = _collect_reasons(classification)
        for name, value in (
            ("val/trusted_top1", trusted_subset_top1(logits_all, targets_all, trusted_mask)),
            ("val/augmentation_agreement", augmentation_agreement(logits_all, strong_logits)),
            ("val/feature_cosine_to_base", drift),
        ):
            if value.value is None and value.reason is not None:
                reasons[name] = value.reason
        return EvaluationResult(
            metrics=result_metrics,
            metric_reasons=reasons,
            per_class_accuracy=classification.per_class_accuracy,
            confusion_matrix=classification.confusion_matrix,
        )


def save_evaluation_artifacts(result: EvaluationResult, output_dir: Path | str) -> None:
    """Save confusion matrix and per-class metrics under `output_dir`.

    Args:
        result: Evaluation result from `Evaluator.evaluate`.
        output_dir: Directory where `confusion_matrix.pt` and
            `per_class_metrics.csv` are written.

    Raises:
        OSError: If files cannot be written.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(result.confusion_matrix, root / "confusion_matrix.pt")
    with (root / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_index", "accuracy", "reason"])
        writer.writeheader()
        for class_index, metric in sorted(result.per_class_accuracy.items()):
            writer.writerow(
                {
                    "class_index": class_index,
                    "accuracy": "" if metric.value is None else f"{metric.value:.12g}",
                    "reason": metric.reason or "",
                }
            )


def _collect_reasons(classification: ClassificationMetrics) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for name, metric in (
        ("val/top1", classification.top1),
        ("val/macro_accuracy", classification.macro_accuracy),
        ("val/bottom_quartile_accuracy", classification.bottom_quartile_accuracy),
    ):
        if metric.value is None and metric.reason is not None:
            reasons[name] = metric.reason
    return reasons


def _get_tensor_attr(output: object, name: str) -> Tensor:
    value = getattr(output, name, None)
    if not isinstance(value, Tensor):
        raise ValueError(f"model output {name!r} must be a torch.Tensor.")
    if not torch.isfinite(value).all():
        raise ValueError(f"model output {name!r} contains NaN or Inf.")
    return value


MappingBySample = dict[str, Tensor]
