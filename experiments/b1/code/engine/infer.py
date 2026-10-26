import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from model_base import load_model
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (block := stream.read(4 * 1024 * 1024)):
            digest.update(block)
    return digest.hexdigest()

def balance(probability, epsilon):
    logp = probability.double().clamp_min(1e-12).log()
    logp = (logp - logp.mean()) / (logp.std(correction=0).clamp_min(1e-08) * epsilon)
    u = torch.zeros(len(logp), dtype=torch.float64, device=logp.device)
    v = torch.zeros(logp.shape[1], dtype=torch.float64, device=logp.device)
    for _ in range(100):
        v = np.log(len(logp) / logp.shape[1]) - torch.logsumexp(logp + u[:, None], 0)
        u = -torch.logsumexp(logp + v[None], 1)
    return (logp + u[:, None] + v[None]).exp().float()

def propagate(seed, index, weights, alpha, iterations):
    coefficient = (alpha * (0.2 + 0.8 * (1 - seed.max(-1).values)))[:, None]
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

def pixels(cache, indices, size, crop=1.0, flip=False):
    array = np.array(cache[indices], copy=True)
    batch = torch.from_numpy(array).cuda().permute(0, 3, 1, 2).float().div_(255)
    side = max(32, round(batch.shape[-1] * crop))
    offset = (batch.shape[-1] - side) // 2
    batch = batch[:, :, offset:offset + side, offset:offset + side]
    batch = F.interpolate(batch, size=(size, size), mode='bicubic', align_corners=False, antialias=True)
    if flip:
        batch = batch.flip(-1)
    return (batch - MEAN.cuda()) / STD.cuda()

def predict(ws, checkpoint, output, cfg, batch_size, split, views, begin=0, end=None):
    cache = {kind: np.load(ROOT / f"image_cache/semi_{split}{('_complete' if kind == 'complete' else '')}_512.npy", mmap_mode='r') for kind in set((v[0] for v in views))}
    original_count = len(next(iter(cache.values())))
    assert all((len(c) == original_count for c in cache.values()))
    end = original_count if end is None else end
    assert 0 <= begin < end <= original_count
    count = end - begin
    model = load_model(ws, cfg, ROOT, initialize_official=False).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['config'] == cfg

    model.load_state_dict(state['model'], strict=True, assign=True)
    model.cuda()
    probability = np.empty((len(views), count, 750), np.float32)
    feature = np.empty((count, 512), np.float32)
    visual = np.empty((count, 768), np.float32)
    captured = {}
    hook = model.backbone.vision_model.post_layernorm.register_forward_hook(lambda module, inputs, output: captured.update(native=output.detach()))
    with torch.inference_mode():
        for start in range(begin, end, batch_size):
            selected = np.arange(start, min(end, start + batch_size))
            local = selected - begin
            embeddings, native_views = ([], [])
            for view_no, (kind, crop, flip) in enumerate(views):
                images = pixels(cache[kind], selected, cfg['size'], crop, flip)
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, embedding = model(images)
                probability[view_no, local] = logits.float().softmax(-1).cpu().numpy()
                embeddings.append(embedding)
                native_views.append(F.normalize(captured['native'].float(), dim=-1))
            visual[local] = F.normalize(torch.stack(native_views).mean(0), dim=-1).cpu().numpy()
            feature[local] = F.normalize(torch.stack(embeddings).mean(0), dim=-1).cpu().numpy()
            if start % (batch_size * 50) == 0:
                print(json.dumps(dict(stage='inference', split=split, images=start, total=count, views=len(views))), flush=True)
    hook.remove()
    assert np.isfinite(visual).all() and visual.shape == (count, 768)
    np.savez(output, probabilities=probability, features=feature, visual=visual, begin=begin, end=end, original_count=original_count)
    del model
    torch.cuda.empty_cache()
