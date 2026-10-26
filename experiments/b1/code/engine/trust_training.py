"""B1 training policies, kept independent of CUDA for unit testing."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def configure_parameters(model, last_blocks):
    layers = model.backbone.vision_model.encoder.layers
    if not 1 <= last_blocks <= len(layers):
        raise ValueError('Invalid number of unfrozen blocks')
    model.backbone.requires_grad_(False)
    for layer in layers[-last_blocks:]:
        layer.requires_grad_(True)
    model.backbone.vision_model.post_layernorm.requires_grad_(True)


def load_quality(root, sha, device):
    root = Path(root)
    path = root / 'cache/train_quality.npz'
    binding = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != binding['quality_sha256']:
        raise ValueError('Quality cache hash mismatch')
    for name, expected in binding['inputs'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Quality input changed: {name}')
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in ['targets', 'trust', 'group_weight']}
    n = len(np.load(root / 'engine/train_labels.npy', allow_pickle=False))
    t, r, w = (arrays[k] for k in ['targets', 'trust', 'group_weight'])
    if (t.shape != (n, 750) or r.shape != (n,) or w.shape != (n,)
            or not all(np.isfinite(a).all() for a in arrays.values())
            or (t < 0).any() or not np.allclose(t.sum(1), 1, atol=1e-5)
            or (r < 0).any() or (r > 1).any() or (w <= 0).any()):
        raise ValueError('Invalid quality arrays')
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}, binding['quality_sha256']


def classification_loss(logits, targets, trust, group_weight, q, floor,
                        permutation=None, coefficient=1.0):
    logp = logits.float().log_softmax(-1)
    def per_row(target):
        # Expected GCE under the soft label, with a soft CE floor.
        return ((1 - floor) * target * (1 - (q * logp).exp()) / q
                - floor * target * logp).sum(-1)
    weight = trust * group_weight
    result = weight * per_row(targets)
    if permutation is not None:
        result = coefficient * result + (1 - coefficient) * weight[permutation] * per_row(targets[permutation])
    # Population-normalized duplicate weights; do not normalize by batch trust.
    return result.mean()


def consistency_loss(left, right, trust, group_weight):
    # Symmetric stop-gradient KL: no second model or test-time adaptation.
    a, b = left.float().log_softmax(-1), right.float().log_softmax(-1)
    loss = .5 * (F.kl_div(a, b.detach().exp(), reduction='none').sum(-1)
                 + F.kl_div(b, a.detach().exp(), reduction='none').sum(-1))
    return (loss * (1 - trust) * group_weight).mean()


def freeze_warmup_gradients(model, step, warmup_steps):
    if step < warmup_steps:
        for name, parameter in model.named_parameters():
            if not (name.startswith('head.') or name == 'log_scale'):
                # None also disables AdamW decay and optimizer state updates.
                parameter.grad = None
