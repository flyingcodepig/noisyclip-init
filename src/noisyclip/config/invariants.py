"""Cross-field configuration invariants for preflight safety."""

from __future__ import annotations

from noisyclip.config.schema import ProjectConfig


def validate_config_invariants(config: ProjectConfig) -> None:
    """Reject internally inconsistent configurations before any data access."""

    lora = config.model.lora
    if lora.enabled:
        if lora.rank <= 0 or lora.alpha <= 0:
            raise ValueError("Enabled LoRA requires positive rank and alpha.")
        if not lora.target_blocks or not lora.target_projections:
            raise ValueError("Enabled LoRA requires target blocks and projections.")
    elif lora.rank != 0 or lora.alpha != 0 or lora.target_blocks or lora.target_projections:
        raise ValueError("Disabled LoRA cannot retain active rank, alpha, or targets.")

    head = config.model.head
    if head.type == "cosine":
        temperatures = (head.temperature_init, head.temperature_min, head.temperature_max)
        if any(item is None for item in temperatures):
            raise ValueError("Cosine head requires initial, minimum, and maximum temperatures.")
        if not head.temperature_min <= head.temperature_init <= head.temperature_max:  # type: ignore[operator]
            raise ValueError("Cosine-head temperature bounds are inconsistent.")

    coefficients = (
        config.noise.signals.ema_loss.coefficient,
        config.noise.signals.augmentation_agreement.coefficient,
        config.noise.signals.prototype_similarity.coefficient,
        config.noise.signals.prototype_margin.coefficient,
        config.noise.signals.prediction_stability.coefficient,
    )
    if config.noise.enabled and sum(coefficients) <= 0:
        raise ValueError("Noise modeling requires at least one positive signal coefficient.")

    partition = config.noise.partition
    if partition.method == "fixed_quantile":
        if partition.trusted_quantile is None or partition.uncertain_quantile is None:
            raise ValueError("Fixed partitioning requires both quantiles.")
        if partition.trusted_quantile >= partition.uncertain_quantile:
            raise ValueError("trusted_quantile must be lower than uncertain_quantile.")

    weights = config.noise.weights
    if weights.uncertain_min > weights.uncertain_max:
        raise ValueError("uncertain_min cannot exceed uncertain_max.")

    if config.loss.feature_anchor.enabled and not config.model.teacher.enabled:
        raise ValueError("Feature anchoring requires the frozen teacher.")
    if config.loss.consistency.enabled and not config.data.strong_transform.enabled:
        raise ValueError("Consistency loss requires the strong transform.")
    if config.loss.elr.enabled and config.loss.elr.weight <= 0:
        raise ValueError("Enabled ELR requires a positive weight.")
    if config.loss.logit_adjustment.enabled and config.loss.logit_adjustment.count_source is None:
        raise ValueError("Logit adjustment requires an explicit trusted count source.")

    feature_cache = config.trainer.frozen_feature_cache
    if feature_cache.enabled:
        if feature_cache.directory is None:
            raise ValueError("Enabled frozen feature cache requires an explicit directory.")
        if not config.model.backbone.freeze or config.model.lora.enabled:
            raise ValueError("Frozen feature cache requires a fully frozen backbone without LoRA.")
        if config.noise.enabled or config.data.strong_transform.enabled:
            raise ValueError("Frozen feature cache is limited to noise-disabled single-view B0/B1.")
        if config.model.teacher.enabled or config.loss.feature_anchor.enabled:
            raise ValueError(
                "Frozen feature cache cannot be used with a teacher or feature anchor."
            )

    reference_cache = config.trainer.reference_feature_cache
    if reference_cache.enabled and reference_cache.directory is None:
        raise ValueError("Enabled reference feature cache requires an explicit directory.")
    if config.evaluation.feature_drift_guard.enabled:
        guard = config.evaluation.feature_drift_guard
        if not config.model.lora.enabled:
            raise ValueError("Feature drift guard is only valid for LoRA adaptation runs.")
        if not reference_cache.enabled:
            raise ValueError("Feature drift guard requires the reference feature cache.")
        if "feature_cosine_to_base" not in config.evaluation.metrics:
            raise ValueError("Feature drift guard requires feature_cosine_to_base evaluation.")
        if guard.catastrophic_minimum_cosine > guard.minimum_cosine:
            raise ValueError(
                "catastrophic_minimum_cosine cannot exceed the diagnostic minimum_cosine."
            )
        if guard.catastrophic_maximum_epoch_drop < guard.maximum_epoch_drop:
            raise ValueError(
                "catastrophic_maximum_epoch_drop cannot be below maximum_epoch_drop."
            )
