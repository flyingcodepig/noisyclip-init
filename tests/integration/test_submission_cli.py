"""Integration tests for F02 submission command-line entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_TEXT = """
experiment: {}
paths: {}
data: {}
model: {}
noise: {}
loss: {}
trainer: {}
evaluation: {}
tracking: {}
submission: {}
"""


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


def write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create config, mapping, and manifest files for CLI tests."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT, encoding="utf-8")
    mapping_path = write_json(tmp_path / "class_to_idx.json", {"0001": 0, "0002": 1})
    manifest_path = write_json(tmp_path / "test_manifest.json", ["a.jpg", "b.jpg"])
    return config_path, mapping_path, manifest_path


def write_package(
    tmp_path: Path,
    *,
    class_to_idx: dict[str, int] | None = None,
    mapping_digest: str | None = None,
    predictions: object | None = None,
) -> Path:
    """Create one fake single-model export package."""

    mapping = class_to_idx or {"0001": 0, "0002": 1}
    if mapping_digest is None:
        from noisyclip.submission.mapping import mapping_digest as compute_digest

        mapping_digest = compute_digest(mapping)
    return write_json(
        tmp_path / "model.json",
        {
            "artifact_type": "noisyclip_single_model_export",
            "models": [{"role": "student"}],
            "class_to_idx": mapping,
            "mapping_digest": mapping_digest,
            "preprocess": {"center_crop": 224, "test_time_augmentation": False},
            "expected_filenames": ["a.jpg", "b.jpg"],
            "predictions": predictions if predictions is not None else [0, 1],
        },
    )


def test_validate_submission_cli_success_and_report(tmp_path: Path) -> None:
    """validate_submission succeeds on a normal headerless CSV and writes JSON."""

    _, mapping_path, manifest_path = write_fixture_files(tmp_path)
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

    _, mapping_path, manifest_path = write_fixture_files(tmp_path)
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


def test_infer_cli_writes_and_validates_prediction_csv(tmp_path: Path) -> None:
    """infer writes pred_results.csv through the package adapter and validates it."""

    config_path, mapping_path, manifest_path = write_fixture_files(tmp_path)
    package_path = write_package(tmp_path)
    output_dir = tmp_path / "out"

    result = run_module(
        "noisyclip.cli.infer",
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
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (output_dir / "pred_results.csv").read_text(encoding="utf-8") == (
        "a.jpg,0001\nb.jpg,0002\n"
    )


def test_infer_cli_rejects_two_models_tta_digest_mismatch_and_existing_output(
    tmp_path: Path,
) -> None:
    """infer rejects multi-model, TTA, mapping digest mismatch, and overwrite attempts."""

    config_path, mapping_path, manifest_path = write_fixture_files(tmp_path)
    package_path = write_package(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "pred_results.csv").write_text("old,0001\n", encoding="utf-8")
    bad_package = write_package(tmp_path, mapping_digest="0" * 64)

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
        "--output-dir",
        str(tmp_path / "tta-out"),
        "--tta",
    )
    digest = run_module(
        "noisyclip.cli.infer",
        "--model",
        str(bad_package),
        "--config",
        str(config_path),
        "--test-manifest",
        str(manifest_path),
        "--class-mapping",
        str(mapping_path),
        "--output-dir",
        str(tmp_path / "digest-out"),
    )
    existing = run_module(
        "noisyclip.cli.infer",
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
    )

    assert two_models.returncode == 3
    assert tta.returncode == 3
    assert digest.returncode == 3
    assert existing.returncode == 3
