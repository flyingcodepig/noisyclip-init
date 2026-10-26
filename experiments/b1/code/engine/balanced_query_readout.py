import argparse, csv, json, os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from infer import ROOT, sha, balance, propagate

def sinkhorn(p, temperature, strength, iterations=120):
    target = p.mean(0).clamp_min(1e-12).pow(1 - strength)
    target /= target.sum()
    kernel = p.clamp_min(1e-12).pow(1 / temperature)
    for _ in range(iterations):
        kernel *= target[None, :] / kernel.mean(0, keepdim=True).clamp_min(1e-30)
        kernel /= kernel.sum(-1, keepdim=True).clamp_min(1e-30)
    assert torch.isfinite(kernel).all() and torch.allclose(kernel.sum(-1), torch.ones(len(kernel), device=p.device), atol=1e-05)
    return kernel
