"""Integration tests for the data audit CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_config(path: Path, train_root: Path, test_root: Path, run_root: Path) -> None:
    path.write_text(
        f"""
experiment: {{}}
paths:
  train_root: "{train_root.as_posix()}"
  test_root: "{test_root.as_posix()}"
  run_root: "{run_root.as_posix()}"
  cache_root: "{(run_root / "cache").as_posix()}"
data:
  expected_num_classes: 3
  val_fraction: 0.34
  split_seed: 123
model: {{}}
noise: {{}}
loss: {{}}
trainer: {{}}
evaluation: {{}}
tracking: {{}}
submission: {{}}
""",
        encoding="utf-8",
    )


def _run_cli(config_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "noisyclip.cli.audit_data", "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_data_audit_cli_succeeds_on_synthetic_three_class_data(tmp_path: Path) -> None:
    """The CLI writes mapping, train/val/test manifests, and summary artifacts."""

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    run_root = tmp_path / "run"
    for class_index, class_id in enumerate(["0001", "0007", "0010"]):
        for image_index in range(3):
            _save_image(
                train_root / class_id / f"{class_id}_{image_index}.png",
                (class_index * 50, image_index * 30, 100),
            )
    _save_image(test_root / "test_a.png", (10, 200, 30))
    _save_image(test_root / "test_b.png", (20, 210, 40))
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_root, test_root, run_root)

    result = _run_cli(config_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert (run_root / "data" / "class_to_idx.json").is_file()
    assert (run_root / "data" / "train_manifest.json").is_file()
    assert (run_root / "data" / "val_manifest.json").is_file()
    assert (run_root / "data" / "test_manifest.json").is_file()
    summary = json.loads((run_root / "data" / "data_summary.json").read_text(encoding="utf-8"))
    assert summary["num_classes"] == 3
    assert summary["num_test"] == 2


def test_data_audit_cli_returns_nonzero_on_leakage(tmp_path: Path) -> None:
    """A test file with the same bytes as a train file fails with code 3."""

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    run_root = tmp_path / "run"
    leaked = train_root / "0001" / "leak.png"
    for class_id in ["0001", "0007", "0010"]:
        _save_image(train_root / class_id / "a.png", (1, int(class_id), 3))
        _save_image(train_root / class_id / "b.png", (2, int(class_id), 4))
    _save_image(leaked, (99, 99, 99))
    test_root.mkdir(parents=True)
    (test_root / "different_name.png").write_bytes(leaked.read_bytes())
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_root, test_root, run_root)

    result = _run_cli(config_path)

    assert result.returncode == 3
    assert "hash intersections" in result.stderr
