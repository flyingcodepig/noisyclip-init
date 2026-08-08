from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

FIELDS = [
    "run_id",
    "experiment_name",
    "seed",
    "status",
    "best_epoch",
    "val_top1",
    "val_macro_accuracy",
    "val_bottom_quartile_accuracy",
    "val_trusted_top1",
    "val_feature_cosine_to_base",
    "peak_gpu_memory_mib",
]


def nested(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    current: Any = data
    for part in key.split("/"):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        metrics_path = run_dir / "metrics" / "best_metrics.json"
        epoch_metrics_path = run_dir / "metrics" / "epoch_metrics.jsonl"
        manifest_path = run_dir / "manifest.json"
        config_path = run_dir / "resolved_config.yaml"
        if not manifest_path.is_file() or not config_path.is_file():
            continue
        epoch_rows = _read_epoch_rows(epoch_metrics_path)
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        elif epoch_rows:
            metrics = max(
                epoch_rows,
                key=lambda row: float(row.get("val/top1", float("-inf"))),
            )
        else:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            continue
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        peak_values = [
            float(row["system/max_gpu_memory_mib"])
            for row in epoch_rows
            if row.get("system/max_gpu_memory_mib") is not None
        ]
        peak_memory = max(peak_values, default=None)
        rows.append(
            {
                "run_id": run_dir.name,
                "experiment_name": config.get("experiment", {}).get("name"),
                "seed": metadata.get("seed"),
                "status": manifest.get("status"),
                "best_epoch": metrics.get("epoch"),
                "val_top1": nested(metrics, "val/top1"),
                "val_macro_accuracy": nested(metrics, "val/macro_accuracy"),
                "val_bottom_quartile_accuracy": nested(metrics, "val/bottom_quartile_accuracy"),
                "val_trusted_top1": nested(metrics, "val/trusted_top1"),
                "val_feature_cosine_to_base": nested(metrics, "val/feature_cosine_to_base"),
                "peak_gpu_memory_mib": peak_memory,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {output}")
    return 0


def _read_epoch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
