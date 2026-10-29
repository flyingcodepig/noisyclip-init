"""Fixed CPU nearest-neighbor propagation with no parameter updates."""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


DEFAULTS = dict(neighbors=24, sharpness=52., compatibility_power=2.6, alpha=.979,
    iterations=8, confidence_power=1.75, local_fraction=.2, local_rank=5,
    nonmutual_fraction=.81, minimum=.58, epsilon=.345,
    assignment_temperature=.96, balance_strength=1., assignment_iterations=124)


def neighbors(geometry, cache, identity):
    """Cache top24 same-encoder query neighbors under an explicit ordered identity."""
    key = hashlib.sha256((identity + hashlib.sha256(geometry.numpy().tobytes()).hexdigest()).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (key + '.npz')
    with target.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            with np.load(target) as archive:
                if str(archive['key'].item()) != key:
                    raise ValueError('Neighbor-cache identity differs')
                index = torch.from_numpy(archive['indices'].copy()).long()
                similarity = torch.from_numpy(archive['similarities'].copy()).float()
        else:
            locations, values = [], []
            for first in range(0, len(geometry), 256):
                last = min(first + 256, len(geometry))
                affinity = geometry[first:last] @ geometry.T
                affinity[torch.arange(last - first), torch.arange(first, last)] = -torch.inf
                value, location = affinity.topk(24, dim=1)
                locations.append(location)
                values.append(value)
            index, similarity = torch.cat(locations), torch.cat(values)
            writing = target.with_name(target.stem + f'.{os.getpid()}.writing.npz')
            np.savez(writing, key=key, indices=index.numpy(), similarities=similarity.numpy())
            writing.replace(target)
    if index.shape != (len(geometry), 24) or not torch.isfinite(similarity).all() or (index == torch.arange(len(geometry))[:, None]).any():
        raise ValueError('Invalid query graph')
    return index, similarity, key


@torch.inference_mode()
def infer(logits, geometry, cache, identity, configuration=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or logits.device.type != 'cpu' or geometry.device.type != 'cpu':
        raise ValueError('Frozen graph inference requires CPU-only inputs and environment')
    config = dict(DEFAULTS)
    if configuration is not None:
        if set(configuration) - set(DEFAULTS):
            raise ValueError('Unknown graph setting')
        config.update(configuration)
    if config['neighbors'] != 24 or logits.shape != (49857, 500) or len(geometry) != len(logits):
        raise ValueError('Unexpected fixed full-TEST shape')
    index, similarity, cache_key = neighbors(geometry, cache, identity)
    energy = (logits.double() / 2.8).softmax(1).clamp_min(1e-12).log()
    energy = (energy - energy.mean()) / (energy.std(correction=0).clamp_min(1e-8) * config['epsilon'])
    row = torch.zeros(len(energy), dtype=torch.float64)
    column = torch.zeros(500, dtype=torch.float64)
    for _ in range(100):
        column = np.log(len(energy) / 500) - torch.logsumexp(energy + row[:, None], 0)
        row = -torch.logsumexp(energy + column[None], 1)
    seed = (energy + row[:, None] + column[None]).exp().float()
    logweight = (similarity - similarity[:, :1]) * config['sharpness']
    distance = (1 - similarity).clamp_min(1e-5)
    radius = distance[:, config['local_rank'] - 1].sqrt()
    local = -distance / (radius[:, None] * radius[index]).clamp_min(1e-8)
    local -= local.max(1, keepdim=True).values
    weight = ((1 - config['local_fraction']) * logweight + config['local_fraction'] * local).exp()
    weight *= seed.max(1).values[index].pow(config['confidence_power'])
    root = seed.sqrt()
    for first in range(0, len(seed), 512):
        last = first + 512
        agreement = (root[first:last, None] * root[index[first:last]]).sum(-1).clamp(0, 1)
        weight[first:last] *= agreement.pow(config['compatibility_power'])
    mutual = (index[index] == torch.arange(len(seed))[:, None, None]).any(-1)
    weight *= config['nonmutual_fraction'] + (1 - config['nonmutual_fraction']) * mutual.float()
    total = weight.sum(1, keepdim=True)
    isolated = total.flatten() <= 0
    normalized = weight / total.clamp_min(1e-12)
    rows = torch.arange(len(seed)).repeat_interleave(24)
    columns, values = index.flatten(), normalized.flatten()
    rows = torch.cat([rows, torch.where(isolated)[0]])
    columns = torch.cat([columns, torch.where(isolated)[0]])
    values = torch.cat([values, torch.ones(int(isolated.sum()))])
    graph = torch.sparse_coo_tensor(torch.stack([rows, columns]), values, (len(seed), len(seed))).coalesce()
    coefficient = config['alpha'] * (config['minimum'] + (1 - config['minimum']) * (1 - seed.max(1, keepdim=True).values))
    posterior = seed
    for _ in range(config['iterations']):
        posterior = (1 - coefficient) * seed + coefficient * torch.sparse.mm(graph, posterior)
    target = posterior.mean(0).clamp_min(1e-12).pow(1 - config['balance_strength'])
    target /= target.sum()
    assigned = posterior.clamp_min(1e-12).pow(1 / config['assignment_temperature'])
    for _ in range(config['assignment_iterations']):
        assigned *= target[None] / assigned.mean(0, keepdim=True).clamp_min(1e-30)
        assigned /= assigned.sum(1, keepdim=True).clamp_min(1e-30)
    for result in [posterior, assigned]:
        if not torch.isfinite(result).all() or not torch.allclose(result.sum(1), torch.ones(len(seed)), atol=1e-5):
            raise ValueError('Nonfinite or unnormalized frozen probabilities')
    return posterior, assigned, dict(configuration=config, neighbor_cache_key=cache_key,
        parameters_updated=False, labels_read=False, class_mass_assumption='equal-column experimental regularization; actual counts unverified')
