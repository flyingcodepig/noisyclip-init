import torch

def propagate_with_floor(seed, index, weights, alpha, iterations, minimum):
    assert 0 <= minimum <= 1 and 0 <= alpha <= 1 and (iterations >= 0)
    coefficient = (alpha * (minimum + (1 - minimum) * (1 - seed.max(-1).values)))[:, None]
    current = seed
    for _ in range(iterations):
        updated = torch.empty_like(current)
        for start in range(0, len(seed), 256):
            ind, weight = (index[start:start + 256], weights[start:start + 256])
            total = weight.sum(-1, keepdim=True)
            mean = (current[ind] * weight[..., None]).sum(1) / total.clamp_min(1e-12)
            updated[start:start + 256] = torch.where(total > 0, mean, seed[start:start + 256])
        current = (1 - coefficient) * seed + coefficient * updated
    assert torch.isfinite(current).all() and torch.allclose(current.sum(-1), torch.ones(len(seed), device=seed.device), atol=1e-05)
    return current
