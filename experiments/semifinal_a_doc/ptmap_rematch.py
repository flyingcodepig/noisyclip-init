"""PT-MAP-style test-only prototype transport; reads no reference labels."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from deliver import sinkhorn
from sweep_rematch_readout import sha256


def balanced_transport(cost, epsilon, steps=60):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    n, classes = cost.shape
    logp = -cost / epsilon
    log_column_target = torch.tensor(n / classes, device=cost.device).log()
    for _ in range(steps):
        logp = logp + log_column_target - torch.logsumexp(logp, dim=0, keepdim=True)
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    return logp.exp()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for full-query transport")
    if not 0 < args.alpha <= 1 or args.iterations < 1:
        raise ValueError("Invalid prototype update schedule")
    cfg = json.loads(Path(args.config).read_text())
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    names = cached["names"]
    if len(names) != cfg["expected_test"] or len(set(names)) != len(names):
        raise ValueError("Wrong fixed test inventory")
    if cached["checkpoint_step"] != cfg["main_steps"] + cfg["tail_steps"]:
        raise ValueError("Expected final checkpoint cache")
    views, original, head = (cached[k].cuda().float() for k in
                             ("view_probabilities", "feature", "head"))
    model_posterior = views.mean(1)
    head_prior = (cfg["head_prior_scale"] * original @ head.T).softmax(-1)
    mixed = (1 - cfg["head_prior_weight"]) * model_posterior + cfg["head_prior_weight"] * head_prior
    balanced_model = sinkhorn(mixed, 0.39, 1, 60)
    minimum = original.min(dim=0, keepdim=True).values
    feature = F.normalize((original - minimum + 1e-6).clamp_min(0).sqrt(), dim=-1)
    centers = balanced_model.T @ feature / balanced_model.sum(dim=0)[:, None]
    fixed_log_posterior = balanced_model.clamp_min(1e-12).log()

    def assignment(mu):
        squared = (feature.square().sum(1)[:, None] + mu.square().sum(1)[None]
                   - 2 * feature @ mu.T).clamp_min(0)
        cost = squared - fixed_log_posterior
        return balanced_transport(cost, args.epsilon)

    for iteration in range(args.iterations):
        transport = assignment(centers)
        estimate = transport.T @ feature / transport.sum(dim=0)[:, None].clamp_min(1e-8)
        centers.lerp_(estimate, args.alpha)
        if iteration in (0, 4, 9, args.iterations - 1):
            print(json.dumps({"iteration": iteration + 1,
                              "column_mass_min": float(transport.sum(0).min()),
                              "column_mass_max": float(transport.sum(0).max())}), flush=True)
    final = assignment(centers)
    prediction = final.argmax(-1).cpu().tolist()
    counts = torch.bincount(torch.tensor(prediction), minlength=cfg["classes"]).float()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pred_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for image_name, label in zip(names, prediction):
            writer.writerow((image_name, f"{label:04d}"))
    record = {"method": "PT-MAP-style posterior-anchored transport",
              "power": 0.5, "min_scope": "per_feature_over_fixed_test_queries",
              "epsilon": args.epsilon, "alpha": args.alpha,
              "iterations": args.iterations, "transport_steps": 60,
              "cost": "squared_l2_minus_log_balanced_model_posterior",
              "cache_sha256": sha256(args.cache), "config_sha256": sha256(args.config),
              "prediction_sha256": sha256(csv_path),
              "class_count_cv": float(counts.std(unbiased=False) / counts.mean()),
              "classes_predicted": int((counts > 0).sum()),
              "class_count_min": int(counts.min()),
              "class_count_max": int(counts.max())}
    (output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
