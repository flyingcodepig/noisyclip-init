from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REQUIRED_FILES = [
    "resolved_config.yaml",
    "manifest.json",
    "data/class_to_idx.json",
    "data/manifest_digest.json",
    "metrics/epoch_metrics.jsonl",
    "metrics/best_metrics.json",
    "metrics/best_eval/per_class_metrics.csv",
    "checkpoints/last.pt",
    "checkpoints/best_top1.pt",
    "artifacts/model.pt",
    "DONE",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (run_dir / relative).is_file()]
    empty = [
        relative
        for relative in REQUIRED_FILES
        if (run_dir / relative).is_file()
        and (run_dir / relative).stat().st_size == 0
        and relative != "DONE"
    ]
    if missing or empty:
        print(json.dumps({"status": "invalid", "missing": missing, "empty": empty}, indent=2))
        return 5

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or (run_dir / "FAILED").exists():
        print(json.dumps({"status": "invalid", "reason": "run is not cleanly COMPLETED"}, indent=2))
        return 5
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        print(json.dumps({"status": "invalid", "reason": "malformed config"}, indent=2))
        return 5
    model = config.get("model", {})
    trainer = config.get("trainer", {})
    lora_enabled = bool(model.get("lora", {}).get("enabled"))
    prototype_enabled = bool(model.get("head", {}).get("prototype_init", {}).get("enabled"))
    conditional: list[str] = []
    if lora_enabled:
        conditional.extend(["metrics/parameter_audit.json", "metrics/lora_merge_equivalence.json"])
        reference = trainer.get("reference_feature_cache", {})
        if not reference.get("enabled"):
            conditional.append("CONFIG:trainer.reference_feature_cache.enabled")
    if prototype_enabled:
        conditional.extend(
            [
                "artifacts/initial_prototypes.pt",
                "artifacts/final_prototypes.pt",
                "artifacts/prototype_initialization.json",
            ]
        )
    missing_conditional = [
        relative
        for relative in conditional
        if relative.startswith("CONFIG:") or not (run_dir / relative).is_file()
    ]
    if missing_conditional:
        print(
            json.dumps({"status": "invalid", "missing_conditional": missing_conditional}, indent=2)
        )
        return 5
    if lora_enabled:
        parameter_report = json.loads(
            (run_dir / "metrics/parameter_audit.json").read_text(encoding="utf-8")
        )
        if parameter_report.get("unexpected_trainable_parameters") != 0:
            print(
                json.dumps(
                    {"status": "invalid", "reason": "unauthorized trainable parameters"},
                    indent=2,
                )
            )
            return 5
        report = json.loads(
            (run_dir / "metrics/lora_merge_equivalence.json").read_text(encoding="utf-8")
        )
        if report.get("valid") is not True:
            print(json.dumps({"status": "invalid", "reason": "LoRA merge failed"}, indent=2))
            return 5
        metric_rows = [
            json.loads(line)
            for line in (run_dir / "metrics/epoch_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if not metric_rows or any(
            not isinstance(row.get("val/feature_cosine_to_base"), int | float)
            for row in metric_rows
        ):
            print(
                json.dumps(
                    {"status": "invalid", "reason": "B2 feature drift metrics are incomplete"},
                    indent=2,
                )
            )
            return 5
    print(json.dumps({"status": "ok", "run_id": run_dir.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
