"""Loss outputs, protocols, and robust loss components."""

from noisyclip.losses.composite import RobustCompositeLoss
from noisyclip.losses.consistency import ConsistencyLoss
from noisyclip.losses.elr import ELRLoss
from noisyclip.losses.feature_anchor import FeatureAnchorLoss
from noisyclip.losses.outputs import CompositeLoss, LossOutput, LossTerm
from noisyclip.losses.weighted_ce import WeightedCrossEntropyLoss

__all__ = [
    "CompositeLoss",
    "ConsistencyLoss",
    "ELRLoss",
    "FeatureAnchorLoss",
    "LossOutput",
    "LossTerm",
    "RobustCompositeLoss",
    "WeightedCrossEntropyLoss",
]
