import math
import numpy as np
import torch

def cutmix(images, permutation, coefficient, rng=np.random):
    h, w = images.shape[-2:]
    ratio = math.sqrt(1 - coefficient)
    bh, bw = (int(h * ratio), int(w * ratio))
    cy, cx = (int(rng.randint(h)), int(rng.randint(w)))
    y1, y2 = (max(0, cy - bh // 2), min(h, cy + (bh + 1) // 2))
    x1, x2 = (max(0, cx - bw // 2), min(w, cx + (bw + 1) // 2))
    result = images.clone()
    result[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    return (result, 1 - (y2 - y1) * (x2 - x1) / (h * w), (y1, y2, x1, x2))
