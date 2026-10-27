"""Synthetic dual-view memory probe only; not a replacement for full-data smoke."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(ROOT / 'engine'))
from single_gpu_support import load_model
from trust_training import configure_parameters, classification_loss, consistency_loss
from trusted_contrast import global_trusted_contrast

p = argparse.ArgumentParser()
p.add_argument('--batch', type=int, required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(41291)
torch.use_deterministic_algorithms(True)
cfg = json.loads((ROOT / 'train_config_a100_stage1.json').read_text())
model = load_model(ROOT, cfg, ROOT).cuda()
configure_parameters(model, 4)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
x = torch.randn(a.batch * 2, 3, 448, 448, device='cuda')
targets = torch.nn.functional.one_hot(torch.arange(a.batch, device='cuda') % 500, 500).float()
trust = torch.full((a.batch,), .5, device='cuda')
weight = torch.ones_like(trust)
torch.cuda.reset_peak_memory_stats()
times = []
for i in range(3):
    start = time.monotonic()
    opt.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
        logits, features = model(x)
    loss = classification_loss(logits[:a.batch], targets, trust, weight, .72, .025)
    loss = loss + .5 * consistency_loss(logits[:a.batch], logits[a.batch:], trust, weight)
    contrast, _ = global_trusted_contrast(features[:a.batch], torch.arange(a.batch, device='cuda') % 10, weight.bool(), .12)
    loss = loss + .031 * contrast
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    assert torch.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    times.append(time.monotonic() - start)
print(json.dumps({'synthetic_memory_probe_only': True, 'micro_batch': a.batch, 'dual_view_forward_batch': 2*a.batch, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'seconds': times}), flush=True)
