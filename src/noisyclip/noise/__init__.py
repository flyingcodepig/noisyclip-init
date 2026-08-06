"""Noise-state records, trust signals, and sample weighting utilities."""

from noisyclip.noise.curriculum import PartitionCurriculum, apply_curriculum
from noisyclip.noise.normalize import ClasswisePercentileNormalizer, percentile_rank_by_class
from noisyclip.noise.partition import apply_partitions, partition_by_class
from noisyclip.noise.pseudolabel import PseudoLabelGate
from noisyclip.noise.signals import (
    AugmentationAgreementSignal,
    EmaLossSignal,
    PredictionStabilitySignal,
    PrototypeMarginSignal,
    PrototypeSimilaritySignal,
)
from noisyclip.noise.state import (
    JsonSampleStateStore,
    PrototypeBuilder,
    SampleState,
    SampleStateStore,
    TrustAggregator,
    TrustSignal,
)
from noisyclip.noise.trust import ClasswiseTrustAggregator

__all__ = [
    "AugmentationAgreementSignal",
    "ClasswisePercentileNormalizer",
    "ClasswiseTrustAggregator",
    "EmaLossSignal",
    "JsonSampleStateStore",
    "PartitionCurriculum",
    "PredictionStabilitySignal",
    "PrototypeBuilder",
    "PrototypeMarginSignal",
    "PrototypeSimilaritySignal",
    "PseudoLabelGate",
    "SampleState",
    "SampleStateStore",
    "TrustAggregator",
    "TrustSignal",
    "apply_curriculum",
    "apply_partitions",
    "partition_by_class",
    "percentile_rank_by_class",
]
