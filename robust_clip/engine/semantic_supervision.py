import torch
from torch.nn import functional as F

def weighted_loss(logits, labels, weights, q, coefficient=1.0, permutation=None, floor=0.0):
    if permutation is not None:
        left = F.cross_entropy(logits, labels, reduction='none')
        right = F.cross_entropy(logits, labels[permutation], reduction='none')
        return (coefficient * weights * left + (1 - coefficient) * weights[permutation] * right).mean()
    logp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    loss = (1 - floor) * (1 - (q * logp).exp()) / q - floor * logp
    return (weights * loss).mean()
