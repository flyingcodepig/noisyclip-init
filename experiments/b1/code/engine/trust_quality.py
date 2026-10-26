"""TRAIN-only group cross-fitting and conservative label-quality estimates."""
import numpy as np
import torch
from torch.nn import functional as F


def grouped_folds(labels, groups, folds, seed):
    """Stratify by each group's label histogram; never split a duplicate group."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if folds < 2 or len(unique) < folds:
        raise ValueError('Need at least two folds and at least one group per fold')
    hist = np.zeros((len(unique), int(labels.max()) + 1))
    np.add.at(hist, (inverse, labels), 1)
    # Each group has total mass one, irrespective of its duplicate count.
    hist /= hist.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-hist[order].max(1), kind='stable')]
    mass = np.zeros((folds, hist.shape[1]))
    sizes = np.zeros(folds)
    assignment = np.empty(len(unique), dtype=np.int64)
    for g in order:
        cost = mass @ hist[g]
        choices = np.flatnonzero(np.isclose(cost, cost.min()))
        f = int(choices[np.argmin(sizes[choices])])
        assignment[g] = f
        mass[f] += hist[g]
        sizes[f] += 1
    result = assignment[inverse]
    if len(np.unique(result)) != folds:
        raise ValueError('Empty fold; reduce folds')
    return result


def crossfit(features, labels, groups, cfg):
    """Ridge heads and kNN use only OTHER folds, one vector per content group.

    No learned feature extractor is fitted on the held-out fold. Input features
    must come from the frozen official CLIP extractor, never the B0 checkpoint.
    """
    device = cfg['device']
    x = F.normalize(torch.as_tensor(features, dtype=torch.float32, device=device), dim=1)
    y = np.asarray(labels, dtype=np.int64)
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    n, classes = len(y), int(y.max()) + 1
    fold = grouped_folds(y, groups, cfg['folds'], cfg['seed'])
    gx = torch.zeros(len(unique), x.shape[1], device=device)
    ids = torch.as_tensor(inverse, device=device)
    gx.index_add_(0, ids, x)
    gx = F.normalize(gx / torch.as_tensor(count, device=device)[:, None], dim=1)
    gy = torch.zeros(len(unique), classes, device=device)
    gy.index_put_((ids, torch.as_tensor(y, device=device)), torch.ones(n, device=device), accumulate=True)
    # Candidate labels are a set, not a majority vote of repeated files.
    gy = (gy > 0).float()
    gy /= gy.sum(1, keepdim=True)
    first = np.unique(inverse, return_index=True)[1]
    gf = fold[first]
    probabilities = np.empty((n, classes), np.float32)
    neighbors = np.empty_like(probabilities)
    missing_classes = []
    for f in range(cfg['folds']):
        train = torch.as_tensor(np.flatnonzero(gf != f), device=device)
        valid = np.flatnonzero(fold == f)
        a, targets = gx[train], gy[train]
        missing_classes.append(torch.where(targets.sum(0) == 0)[0].cpu().tolist())
        reg = cfg['ridge'] * torch.eye(a.shape[1], device=device)
        weights = torch.linalg.solve(a.T @ a + reg, a.T @ targets)
        k = min(cfg['neighbors'], len(train))
        for start in range(0, len(valid), cfg['batch']):
            rows = valid[start:start + cfg['batch']]
            q = x[rows]
            probabilities[rows] = (q @ weights / cfg['temperature']).softmax(-1).cpu().numpy()
            scores, indices = (q @ a.T).topk(k, dim=1)
            vote = scores.div(cfg['neighbor_temperature']).softmax(-1)
            neighbors[rows] = (targets[indices] * vote[..., None]).sum(1).cpu().numpy()
    return probabilities, neighbors, fold, missing_classes


def make_targets(probability, neighbors, labels, groups, cfg):
    """Return dense soft targets plus label trust and duplicate weights.

    New labels receive mass only when OOF head and OOF neighbors agree.
    Unsupported/tail examples retain a nonzero original candidate component.
    """
    p, v = np.asarray(probability), np.asarray(neighbors)
    y = np.asarray(labels)
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    candidates = np.zeros((len(count), p.shape[1]), np.float32)
    candidates[inverse, y] = 1
    candidates /= candidates.sum(1, keepdims=True)
    original = candidates[inverse]
    evidence = (p + v) / 2
    support = (evidence * (original > 0)).sum(1)
    reference = np.empty(len(y), np.float32)
    for c in np.unique(y):
        rows = y == c
        reference[rows] = max(float(np.median(support[rows])), 1e-6)
    relative = np.minimum(support / reference, 1)
    # Absolute support prevents a uniformly poor class from becoming all trusted.
    trust = np.clip(np.sqrt(relative * support), cfg['minimum_trust'], 1).astype(np.float32)
    agree = p.argmax(1) == v.argmax(1)
    confident = evidence.max(1) >= cfg['correction_threshold']
    correction = (1 - trust) * agree * confident
    soft = (1 - correction[:, None]) * original + correction[:, None] * evidence
    soft /= soft.sum(1, keepdims=True)
    group_weight = (1 / count[inverse]).astype(np.float32)
    group_weight /= group_weight.mean()
    return soft.astype(np.float32), trust, group_weight
