"""Fixed-model four-view probability weights; no labels are read here."""
import argparse
import csv
import json
from pathlib import Path

import torch

from deliver import graph, sinkhorn
from sweep_rematch_readout import propagate, sha256


WEIGHTS = {
    "documented": (0.23, 0.16, 0.21, 0.40),
    "equal": (0.25, 0.25, 0.25, 0.25),
    "center_focus": (0.15, 0.15, 0.25, 0.45),
    "full_focus": (0.40, 0.20, 0.20, 0.20),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    names = cached["names"]
    if len(names) != cfg["expected_test"] or len(set(names)) != len(names):
        raise ValueError("Wrong cached test inventory")
    if cached["checkpoint_step"] != cfg["main_steps"] + cfg["tail_steps"]:
        raise ValueError("Cache checkpoint is not final")
    view_probs, feature, visual, head = (cached[k].cuda() for k in
                                       ("view_probabilities", "feature", "visual", "head"))
    prior = (cfg["head_prior_scale"] * feature @ head.T).softmax(-1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"cache_sha256": sha256(args.cache), "config_sha256": sha256(args.config), "variants": {}}
    for name, weights in WEIGHTS.items():
        if abs(sum(weights) - 1) > 1e-8:
            raise ValueError("Weights must sum to one")
        vector = torch.tensor(weights, device="cuda")
        prob = (view_probs * vector[None, :, None]).sum(1)
        mixed = (1 - cfg["head_prior_weight"]) * prob + cfg["head_prior_weight"] * prior
        seed = sinkhorn(mixed, 0.39, 1, 60)
        edges = [graph(x, seed, cfg["neighbors"]) for x in (feature, visual)]
        alpha = cfg["graph_alpha"] * (0.63 + 0.37 * (1 - seed.max(-1).values))[:, None]
        current = propagate(seed, edges, alpha, True)
        prediction = sinkhorn(current, 0.78, 1, 100).argmax(-1).cpu().tolist()
        counts = torch.bincount(torch.tensor(prediction), minlength=cfg["classes"]).float()
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for image_name, label in zip(names, prediction):
                writer.writerow((image_name, f"{label:04d}"))
        manifest["variants"][name] = {
            "weights": weights,
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
