"""Actual-dataset adapters, isolated from original 750-class B1 behavior."""
import json
import numpy as np
import torch
from model_base import load_model as original_load_model


def load_model(workspace, config, root, initialize_official=True):
    model = original_load_model(workspace, config, root, initialize_official)
    model.head = torch.nn.Linear(512, config['num_classes'], bias=False)
    return model


def load_actual_quality(root, sha, device, classes):
    path = root / 'cache/a100/train_quality.npz'
    binding = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != binding['quality_sha256']:
        raise ValueError('Actual-dataset quality hash mismatch')
    for name, expected in binding['inputs'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Actual-dataset input changed: {name}')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ['targets', 'trust', 'group_weight']}
    labels = np.load(root / 'cache/a100/train_labels.npy', allow_pickle=False)
    t, r, w = (arrays[k] for k in ['targets', 'trust', 'group_weight'])
    if (t.shape != (len(labels), classes) or r.shape != labels.shape or w.shape != labels.shape
            or not all(np.isfinite(a).all() for a in arrays.values())
            or (t < 0).any() or not np.allclose(t.sum(1), 1, atol=1e-5)
            or (r < .1 - 1e-6).any() or (r > 1).any() or (w <= 0).any()):
        raise ValueError('Invalid actual-dataset quality arrays')
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}, binding['quality_sha256']
