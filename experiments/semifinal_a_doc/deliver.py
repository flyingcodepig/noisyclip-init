"""Four-view, one-model E1 readout and submission packaging."""
import argparse
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from data import inventory, open_rgb, view
from model import AModel


class Query(Dataset):
    def __init__(self, rows, specs): self.rows, self.specs = rows, specs
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        image = open_rgb(self.rows[index][0])
        return torch.stack([view(image, kind, crop, flip) for kind, crop, flip, _ in self.specs]), index


def sinkhorn(probability, temperature, strength, iterations):
    logp = probability.clamp_min(1e-12).log().double() / temperature
    target = probability.double().mean(0).clamp_min(1e-12).pow(1 - strength)
    target /= target.sum()
    log_target = target.log()
    for _ in range(iterations):
        logp = logp + (log_target - torch.logsumexp(logp, 0) + np.log(len(logp)))
        logp = logp - torch.logsumexp(logp, 1, keepdim=True)
    return logp.exp().float()


def knn(feature, neighbors, block=256):
    feature = F.normalize(feature.float(), dim=-1)
    indexes, similarities = [], []
    for start in range(0, len(feature), block):
        end = min(start + block, len(feature))
        similarity = feature[start:end] @ feature.T
        similarity[torch.arange(end - start, device=feature.device), torch.arange(start, end, device=feature.device)] = -2
        value, index = similarity.topk(neighbors, -1)
        indexes.append(index); similarities.append(value)
    return torch.cat(indexes), torch.cat(similarities)


def graph(feature, seed, neighbors):
    index, sim = knn(feature, neighbors)
    n, k = index.shape
    # Mutual-neighbor indicator is computed in bounded batches.
    mutual = torch.zeros_like(sim, dtype=torch.bool)
    rows = torch.arange(n, device=index.device)
    for j in range(k):
        mutual[:, j] = (index[index[:, j]] == rows[:, None]).any(-1)
    weight = torch.exp((sim - 1) / 0.07) * seed.max(-1).values[index] * sim.clamp_min(1e-6) * (0.2 + 0.8 * mutual.float())
    weight /= weight.sum(-1, keepdim=True).clamp_min(1e-20)
    return index, weight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", required=True)
    p.add_argument("--official-weights", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if state["step"] != cfg["main_steps"] + cfg["tail_steps"]: raise ValueError("Expected final checkpoint")
    model = AModel(args.official_weights, cfg["classes"], cfg["image_size"])
    model.load_state_dict(state["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    rows = inventory(args.test_root, train=False)
    loader = DataLoader(Query(rows, cfg["views"]), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    probs, features, visuals = [], [], []
    weights = torch.tensor([v[3] for v in cfg["views"]], device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, (images, _) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            view_probs, view_features, view_visuals = [], [], []
            for i in range(len(cfg["views"])):
                logits, feature, visual = model(images[:, i])
                view_probs.append((logits.float() / cfg["posterior_temperature"]).softmax(-1))
                view_features.append(feature.float()); view_visuals.append(visual.float())
            prob = (torch.stack(view_probs) * weights[:, None, None]).sum(0)
            feature = F.normalize(torch.stack(view_features).mean(0), dim=-1)
            visual = F.normalize(torch.stack(view_visuals).mean(0), dim=-1)
            probs.append(prob.cpu()); features.append(feature.cpu()); visuals.append(visual.cpu())
            if number % 50 == 0: print(f"inference {number * args.batch}/{len(rows)}", flush=True)
    device = "cuda"
    prob = torch.cat(probs).to(device)
    feature = torch.cat(features).to(device)
    visual = torch.cat(visuals).to(device)
    head = F.normalize(model.head.weight.float(), dim=-1)
    prior = (cfg["head_prior_scale"] * feature @ head.T).softmax(-1)
    mixed = (1 - cfg["head_prior_weight"]) * prob + cfg["head_prior_weight"] * prior
    seed = sinkhorn(mixed, 0.39, 1, 60)
    edges = [graph(x, seed, cfg["neighbors"]) for x in (feature, visual)]
    alpha = cfg["graph_alpha"] * (0.63 + 0.37 * (1 - seed.max(-1).values))[:, None]
    current = seed
    for iteration in range(1, cfg["graph_iterations"] + 1):
        aggregate = torch.zeros_like(current)
        for index, weight in edges:
            for start in range(0, len(seed), 256):
                part = slice(start, start + 256)
                aggregate[part] += 0.5 * (current[index[part]] * weight[part, :, None]).sum(1)
        current = (1 - alpha) * seed + alpha * aggregate
        current /= current.sum(-1, keepdim=True).clamp_min(1e-12)
        if iteration in (2, 4, 6): current = sinkhorn(current, 0.90, 0.60, 30)
    final = sinkhorn(current, 0.78, 1, 100).argmax(-1).cpu().tolist()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pred_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for (path, _), label in zip(rows, final): writer.writerow((Path(path).name, f"{label:04d}"))
    with zipfile.ZipFile(output / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    print(json.dumps({"images": len(rows), "classes_predicted": len(set(final)), "zip": str(output / "submission.zip")}))


if __name__ == "__main__": main()
