"""Regression coverage for formal run validation and result collection scripts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "init_build/04_scripts_and_configs/scripts/check_run_artifacts.py"
COLLECTOR = ROOT / "init_build/04_scripts_and_configs/scripts/collect_results.py"


def test_checker_accepts_current_b2_artifact_layout(tmp_path: Path) -> None:
    """The checker recognizes COMPLETED plus B2-specific evidence."""

    run = _write_run(tmp_path / "b2-run", lora=True, prototypes=True)
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(run)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "ok"


def test_collector_reads_epoch_jsonl_and_nested_manifest(tmp_path: Path) -> None:
    """Result collection works with actual epoch JSONL and manifest metadata."""

    root = tmp_path / "runs"
    run = _write_run(root / "b2-run", lora=False, prototypes=False)
    (run / "metrics/best_metrics.json").unlink()
    output = tmp_path / "summary.csv"
    result = subprocess.run(
        [sys.executable, str(COLLECTOR), "--run-root", str(root), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "fixture"
    assert rows[0]["seed"] == "7"
    assert rows[0]["status"] == "COMPLETED"
    assert rows[0]["best_epoch"] == "1"
    assert rows[0]["val_top1"] == "0.6"
    assert rows[0]["peak_gpu_memory_mib"] == "12.0"


def _write_run(path: Path, *, lora: bool, prototypes: bool) -> Path:
    required = (
        "data",
        "metrics/best_eval",
        "checkpoints",
        "artifacts",
    )
    for relative in required:
        (path / relative).mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": {"name": "fixture"},
        "model": {
            "lora": {"enabled": lora},
            "head": {"prototype_init": {"enabled": prototypes}},
        },
        "trainer": {"reference_feature_cache": {"enabled": lora}},
    }
    (path / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (path / "manifest.json").write_text(
        json.dumps({"status": "COMPLETED", "metadata": {"seed": 7}}), encoding="utf-8"
    )
    rows = [
        {
            "epoch": 0,
            "val/top1": 0.5,
            "val/macro_accuracy": 0.4,
            "val/feature_cosine_to_base": 0.99,
            "system/max_gpu_memory_mib": 10.0,
        },
        {
            "epoch": 1,
            "val/top1": 0.6,
            "val/macro_accuracy": 0.5,
            "val/feature_cosine_to_base": 0.98,
            "system/max_gpu_memory_mib": 12.0,
        },
    ]
    (path / "metrics/epoch_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "metrics/best_metrics.json").write_text(json.dumps(rows[-1]), encoding="utf-8")
    for relative in (
        "data/class_to_idx.json",
        "data/manifest_digest.json",
        "metrics/best_eval/per_class_metrics.csv",
        "checkpoints/last.pt",
        "checkpoints/best_top1.pt",
        "artifacts/model.pt",
    ):
        (path / relative).write_bytes(b"x")
    (path / "DONE").write_text("done\n", encoding="utf-8")
    if lora:
        (path / "metrics/parameter_audit.json").write_text(
            json.dumps({"unexpected_trainable_parameters": 0}), encoding="utf-8"
        )
        (path / "metrics/lora_merge_equivalence.json").write_text(
            json.dumps({"valid": True}), encoding="utf-8"
        )
    if prototypes:
        (path / "artifacts/initial_prototypes.pt").write_bytes(b"x")
        (path / "artifacts/final_prototypes.pt").write_bytes(b"x")
        (path / "artifacts/prototype_initialization.json").write_text("{}\n", encoding="utf-8")
    return path
