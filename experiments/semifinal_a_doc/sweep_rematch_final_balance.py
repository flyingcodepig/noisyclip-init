"""Label-free final balancing sweep using one frozen E1 graph state."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from deliver import graph, sinkhorn
from sweep_rematch_readout import propagate, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    names = cached["names"]
    if len(names) != cfg["expected_test"]:
        raise ValueError("Wrong cached test size")
    prob, feature, visual, head = (cached[k].cuda() for k in ("prob", "feature", "visual", "head"))
    prior = (cfg["head_prior_scale"] * feature @ head.T).softmax(-1)
    mixed = (1 - cfg["head_prior_weight"]) * prob + cfg["head_prior_weight"] * prior
    seed = sinkhorn(mixed, 0.39, 1, 60)
    edges = [graph(x, seed, cfg["neighbors"]) for x in (feature, visual)]
    alpha = cfg["graph_alpha"] * (0.63 + 0.37 * (1 - seed.max(-1).values))[:, None]
    current = propagate(seed, edges, alpha, True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"cache_sha256": sha256(args.cache), "config_sha256": sha256(args.config), "variants": {}}
    for temperature in (0.65, 0.72, 0.78, 0.85, 0.92):
        for strength in (0.8, 0.9, 1.0, 1.1):
            name = f"temp{temperature:g}_strength{strength:g}"
            labels = sinkhorn(current, temperature, strength, 100).argmax(-1).cpu().tolist()
            counts = torch.bincount(torch.tensor(labels), minlength=cfg["classes"]).float()
            path = output / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                for image_name, label in zip(names, labels):
                    writer.writerow((image_name, f"{label:04d}"))
            manifest["variants"][name] = {
                "csv_sha256": sha256(path),
                "class_count_cv": float(counts.std(unbiased=False) / counts.mean()),
                "classes_predicted": int((counts > 0).sum()),
                "class_count_min": int(counts.min()),
                "class_count_max": int(counts.max()),
            }
            print(name, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
