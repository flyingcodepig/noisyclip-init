"""Frozen-model, label-free readout diagnostics from a four-view cache."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from deliver import graph, sinkhorn


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def propagate(seed, edges, alpha, mid_balance):
    current = seed
    for iteration in range(1, 9):
        aggregate = torch.zeros_like(current)
        for index, weight in edges:
            for start in range(0, len(seed), 256):
                part = slice(start, start + 256)
                aggregate[part] += (current[index[part]] * weight[part, :, None]).sum(1) / len(edges)
        current = (1 - alpha) * seed + alpha * aggregate
        current /= current.sum(-1, keepdim=True).clamp_min(1e-12)
        if mid_balance and iteration in (2, 4, 6):
            current = sinkhorn(current, 0.90, 0.60, 30)
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for graph readout")
    cfg = json.loads(Path(args.config).read_text())
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    names = cached["names"]
    if len(names) != cfg["expected_test"] or len(set(names)) != len(names):
        raise ValueError("Invalid cache names")
    if cached["checkpoint_step"] != cfg["main_steps"] + cfg["tail_steps"]:
        raise ValueError("Cache was not made from the final checkpoint")
    prob = cached["prob"].cuda()
    feature = cached["feature"].cuda()
    visual = cached["visual"].cuda()
    head = cached["head"].cuda()
    prior = (cfg["head_prior_scale"] * feature @ head.T).softmax(-1)
    mixed = (1 - cfg["head_prior_weight"]) * prob + cfg["head_prior_weight"] * prior
    seed = sinkhorn(mixed, 0.39, 1, 60)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"cache_sha256": sha256(args.cache), "config_sha256": sha256(args.config), "variants": {}}

    def save(name, prediction):
        labels = prediction.argmax(-1).cpu().tolist()
        counts = torch.bincount(torch.tensor(labels), minlength=cfg["classes"]).float()
        csv_path = output / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for image_name, label in zip(names, labels):
                writer.writerow((image_name, f"{label:04d}"))
        manifest["variants"][name] = {
            "csv_sha256": sha256(csv_path),
            "classes_predicted": int((counts > 0).sum()),
            "class_count_cv": float(counts.std(unbiased=False) / counts.mean()),
            "class_count_min": int(counts.min()),
            "class_count_max": int(counts.max()),
        }
        print(name, json.dumps(manifest["variants"][name]), flush=True)

    save("direct", prob)
    save("prior_mixed", mixed)
    save("seed_balanced", seed)
    alpha = cfg["graph_alpha"] * (0.63 + 0.37 * (1 - seed.max(-1).values))[:, None]
    for mode in ("e1", "joint"):
        if mode == "e1":
            edges = [graph(x, seed, cfg["neighbors"]) for x in (feature, visual)]
        else:
            combined = F.normalize(torch.cat((feature, visual), dim=-1), dim=-1)
            edges = [graph(combined, seed, cfg["neighbors"])]
        for mid in (False, True):
            current = propagate(seed, edges, alpha, mid)
            prefix = f"{mode}_{'mid' if mid else 'nomid'}"
            save(f"{prefix}_nofinal", current)
            for strength in (0.0, 0.5, 1.0):
                save(f"{prefix}_final{strength:g}", sinkhorn(current, 0.78, strength, 100))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
