import torch
from torch.nn import functional as F

def global_trusted_contrast(features, labels, trusted, temperature):
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    if distributed:
        from torch.distributed.nn.functional import all_gather
        gathered = torch.cat(all_gather(features), 0)
        label_parts = [torch.empty_like(labels) for _ in range(world)]
        trust_parts = [torch.empty_like(trusted) for _ in range(world)]
        torch.distributed.all_gather(label_parts, labels)
        torch.distributed.all_gather(trust_parts, trusted)
        all_labels, all_trusted = (torch.cat(label_parts), torch.cat(trust_parts))
    else:
        gathered, all_labels, all_trusted = (features, labels, trusted)
    n = len(labels)
    self_mask = torch.zeros(n, len(gathered), dtype=torch.bool, device=features.device)
    self_mask[torch.arange(n, device=features.device), rank * n + torch.arange(n, device=features.device)] = True
    positive = (labels[:, None] == all_labels[None, :]) & trusted[:, None] & all_trusted[None, :] & ~self_mask
    count = positive.sum(-1)
    logp = (features @ gathered.T / temperature).masked_fill(self_mask, -10000.0).log_softmax(-1)
    row_loss = -(logp * positive).sum(-1) / count.clamp_min(1)
    valid = count > 0
    denominator = valid.sum().detach().clone()
    if distributed:
        torch.distributed.all_reduce(denominator)
    loss = world * (row_loss * valid).sum() / denominator.clamp_min(1)
    return (loss, dict(local_positive_pairs=int(positive.sum()), local_positive_rows=int(valid.sum())))
