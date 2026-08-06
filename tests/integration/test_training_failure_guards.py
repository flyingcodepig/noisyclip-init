"""Training failure guard integration tests."""

from __future__ import annotations

import json

import pytest
import torch
from test_two_batch_train import tiny_components

from noisyclip.config.loader import load_config_from_mapping
from noisyclip.engine.trainer import Trainer, TrainingFailedError, TrainingPreflightError
from noisyclip.tracking.artifacts import create_run_dir


def test_existing_run_directory_refuses_overwrite(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Tracking refuses old run directories when fail_if_run_exists is true."""

    create_run_dir(tmp_path, "run")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "run", fail_if_run_exists=True)


def test_nan_loss_marks_failed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """NaN loss enters FAILED and leaves diagnostics."""

    config, components = tiny_components(tmp_path)
    components.loss = _NanLoss()
    with pytest.raises(TrainingFailedError, match="NaN"):
        Trainer(config=config, components=components, device="cpu").fit()
    assert (tmp_path / "run" / "FAILED").is_file()


def test_all_losses_disabled_fails_preflight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RobustCompositeLoss-style all-disabled config never reaches backward."""

    config = load_config_from_mapping(
        {
            "experiment": {},
            "paths": {"run_root": "runs"},
            "data": {"expected_num_classes": 3},
            "model": {},
            "noise": {},
            "loss": {
                "cross_entropy": {"enabled": False, "weight": 0.0},
                "elr": {"enabled": False, "weight": 0.0},
                "consistency": {"enabled": False, "weight": 0.0},
                "feature_anchor": {"enabled": False, "weight": 0.0},
            },
            "trainer": {"device": "cpu", "precision": "fp32"},
            "evaluation": {},
            "tracking": {"minimum_free_disk_gib": 0.000000001},
            "submission": {},
        }
    )
    _, components = tiny_components(tmp_path)
    with pytest.raises(TrainingPreflightError, match="At least one training loss"):
        Trainer(config=config, components=components, device="cpu").preflight()


def test_trainable_parameter_guards_detect_unauthorized_backbone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Actual trainable parameter set catches accidental backbone updates."""

    config, components = tiny_components(tmp_path)
    for parameter in components.model.backbone.parameters():
        parameter.requires_grad = True
    with pytest.raises(TrainingPreflightError, match="Unauthorized trainable"):
        Trainer(config=config, components=components, device="cpu").preflight()


def test_corrupt_committed_sample_state_is_not_silently_reset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A malformed prior state must fail instead of reverting to default trust."""

    config, components = tiny_components(tmp_path)
    manifest = components.artifact_store.sample_state_dir() / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "epoch": 0, "state_file": "missing.json"}),
        encoding="utf-8",
    )

    with pytest.raises(TrainingFailedError, match="unsafe or inconsistent"):
        Trainer(config=config, components=components, device="cpu").fit()


class _NanLoss:
    """Loss fake that returns NaN to exercise failure guards."""

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        from noisyclip.losses.outputs import LossOutput

        total = torch.tensor(float("nan"), requires_grad=True)
        return LossOutput(total=total, components={"loss/nan": total}, per_sample_supervised=None)
