"""Integration tests for F02 submission command-line entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from noisyclip.submission.validator import ValidationReport

ROOT = Path(__file__).resolve().parents[2]


def run_module(module: str, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run a NoisyCLIP CLI module with the local source tree on PYTHONPATH."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def write_json(path: Path, payload: object) -> Path:
    """Write UTF-8 JSON test data and return its path."""

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create config, mapping, and manifest files for CLI tests."""

    test_root = tmp_path / "test"
    test_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment: {{}}
paths:
  test_root: "{test_root.as_posix()}"
  cache_root: "{(tmp_path / "cache").as_posix()}"
data: {{}}
model: {{}}
noise: {{}}
loss: {{}}
trainer:
  device: cpu
  batch_size: 2
  num_workers: 0
evaluation: {{}}
tracking: {{}}
submission: {{}}
""",
        encoding="utf-8",
    )
    mapping_path = write_json(tmp_path / "class_to_idx.json", {"0001": 0, "0002": 1})
    manifest_path = write_json(tmp_path / "test_manifest.json", ["a.jpg", "b.jpg"])
    return config_path, mapping_path, manifest_path, test_root


def test_validate_submission_cli_success_and_report(tmp_path: Path) -> None:
    """validate_submission succeeds on a normal headerless CSV and writes JSON."""

    _, mapping_path, manifest_path, _ = write_fixture_files(tmp_path)
    csv_path = tmp_path / "pred_results.csv"
    csv_path.write_text("a.jpg,0001\nb.jpg,0002\n", encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_module(
        "noisyclip.cli.validate_submission",
        "--csv",
        str(csv_path),
        "--test-manifest",
        str(manifest_path),
        "--class-mapping",
        str(mapping_path),
        "--report-json",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True


def test_validate_submission_cli_returns_five_for_bad_artifact(tmp_path: Path) -> None:
    """validate_submission returns 5 when `1` is used instead of `0001`."""

    _, mapping_path, manifest_path, _ = write_fixture_files(tmp_path)
    csv_path = tmp_path / "pred_results.csv"
    csv_path.write_text("a.jpg,1\nb.jpg,0002\n", encoding="utf-8")

    result = run_module(
        "noisyclip.cli.validate_submission",
        "--csv",
        str(csv_path),
        "--test-manifest",
        str(manifest_path),
        "--class-mapping",
        str(mapping_path),
    )

    assert result.returncode == 5
    assert "CLASS_ID_FORMAT" in result.stdout


def test_infer_cli_wires_single_model_inference(tmp_path: Path, monkeypatch, capsys) -> None:
    """infer forwards one model and resolved data paths to the inference library."""

    from noisyclip.cli import infer

    config_path, mapping_path, manifest_path, test_root = write_fixture_files(tmp_path)
    package_path = tmp_path / "model.pt"
    package_path.touch()
    output_dir = tmp_path / "out"

    def fake_inference(*args, **kwargs):
        output_dir.mkdir()
        (output_dir / "pred_results.csv").write_text("a.jpg,0001\nb.jpg,0002\n", encoding="utf-8")
        assert args == (package_path, output_dir)
        assert kwargs["test_manifest_path"] == manifest_path
        assert kwargs["test_root"] == test_root
        assert kwargs["class_mapping_path"] == mapping_path
        assert kwargs["device"] == "cpu"
        return ValidationReport(valid=True, row_count=2, expected_count=2, issues=())

    monkeypatch.setattr(infer, "run_packaged_submission_inference", fake_inference)

    result = infer.main(
        [
            "--model",
            str(package_path),
            "--config",
            str(config_path),
            "--test-manifest",
            str(manifest_path),
            "--class-mapping",
            str(mapping_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert "SUBMISSION_OK" in capsys.readouterr().out
    assert (output_dir / "pred_results.csv").read_text(encoding="utf-8") == (
        "a.jpg,0001\nb.jpg,0002\n"
    )


def test_infer_cli_rejects_two_models_and_tta(
    tmp_path: Path,
) -> None:
    """infer rejects multi-model and test-time augmentation requests."""

    config_path, mapping_path, manifest_path, test_root = write_fixture_files(tmp_path)
    package_path = tmp_path / "model.pt"
    package_path.touch()

    two_models = run_module(
        "noisyclip.cli.infer",
        "--model",
        str(package_path),
        "--model",
        str(package_path),
        "--config",
        str(config_path),
        "--test-manifest",
        str(manifest_path),
        "--class-mapping",
        str(mapping_path),
        "--test-root",
        str(test_root),
        "--output-dir",
        str(tmp_path / "other-out"),
    )
    tta = run_module(
        "noisyclip.cli.infer",
        "--model",
        str(package_path),
        "--config",
        str(config_path),
        "--test-manifest",
        str(manifest_path),
        "--class-mapping",
        str(mapping_path),
        "--test-root",
        str(test_root),
        "--output-dir",
        str(tmp_path / "tta-out"),
        "--tta",
    )
    assert two_models.returncode == 3
    assert tta.returncode == 3
