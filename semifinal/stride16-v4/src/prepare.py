"""Extract both frozen official CLIP views and construct train-only prototypes."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModelWithProjection
from data import inventory, open_rgb, view, save_inventory


class TwoViews(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        image = open_rgb(self.rows[i][0])
        return view(image, "full", size=224), view(image, "center", size=224)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", required=True)
    p.add_argument("--official-weights", required=True, help="Local official OpenAI CLIP ViT-B/32 HF-compatible folder")
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--classes", type=int, default=500)
    p.add_argument("--expected-train", type=int, default=90197)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = inventory(args.train_root, classes=args.classes, expected=args.expected_train)
    save_inventory(rows, out / "train_inventory.json")
    model = CLIPVisionModelWithProjection.from_pretrained(args.official_weights, local_files_only=True).cuda().eval()
    features = [[], []]
    loader = DataLoader(TwoViews(rows), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for n, (full, center) in enumerate(loader):
            for slot, pixels in enumerate((full, center)):
                f = F.normalize(model(pixels.cuda(non_blocking=True)).image_embeds.float(), dim=-1)
                features[slot].append(f.cpu().numpy())
            if n % 50 == 0: print(f"teacher {n * args.batch}/{len(rows)}", flush=True)
    full, center = (np.concatenate(x).astype(np.float32) for x in features)
    teacher = full + center
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True).clip(1e-8)
    labels = np.array([r[1] for r in rows], np.int64)
    canonical_map = {}
    canonical = np.empty(len(rows), np.int32)
    for i, (path, _) in enumerate(rows):
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        canonical[i] = canonical_map.setdefault(digest, len(canonical_map))
    sums = np.zeros((args.classes, 512), np.float32)
    np.add.at(sums, labels, teacher)
    counts = np.bincount(labels, minlength=args.classes)
    centers = sums / np.linalg.norm(sums, axis=1, keepdims=True).clip(1e-8)
    # Original-label leave-one-out decision and two-view high-margin candidate rule.
    own = sums[labels] - teacher
    own /= np.linalg.norm(own, axis=1, keepdims=True).clip(1e-8)
    loo = teacher @ centers.T
    loo[np.arange(len(labels)), labels] = (teacher * own).sum(-1)
    loo_agree = loo.argmax(-1) == labels
    pf, pc = full @ centers.T, center @ centers.T
    tf, tc = np.argsort(pf, axis=1)[:, -2:], np.argsort(pc, axis=1)[:, -2:]
    candidate = np.full((len(labels), 2), -1, np.int16)
    candidate[:, 0] = labels
    agreed = (tf[:, 1] == tc[:, 1]) & (tf[:, 1] != labels)
    margin = (np.take_along_axis(pf, tf[:, 1:], 1) - np.take_along_axis(pf, tf[:, :1], 1)).ravel()
    margin2 = (np.take_along_axis(pc, tc[:, 1:], 1) - np.take_along_axis(pc, tc[:, :1], 1)).ravel()
    candidate[agreed & (margin >= 0.03) & (margin2 >= 0.03), 1] = tf[agreed & (margin >= 0.03) & (margin2 >= 0.03), 1]
    np.savez(out / "train_inputs.npz", teacher=teacher.astype(np.float16), labels=labels.astype(np.int16),
             centers=centers, candidate=candidate, loo_agree=loo_agree, canonical=canonical)
    print(json.dumps({"images": len(rows), "extra_candidates": int((candidate[:, 1] >= 0).sum())}))


if __name__ == "__main__": main()
