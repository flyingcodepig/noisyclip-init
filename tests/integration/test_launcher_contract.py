"""Deployment launcher-to-CLI contract regression tests."""

from __future__ import annotations

from pathlib import Path

from noisyclip.cli.train import build_parser
from noisyclip.config.loader import load_config

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "init_build" / "04_scripts_and_configs" / "scripts" / "run_experiment.sh"
B0_CONFIG = (
    ROOT
    / "init_build"
    / "04_scripts_and_configs"
    / "configs"
    / "experiments"
    / "b0_frozen_linear.yaml"
)


def test_launcher_only_passes_supported_train_cli_arguments() -> None:
    """The shell launcher must not pass flags absent from the train parser."""

    launcher = LAUNCHER.read_text(encoding="utf-8")
    parser = build_parser()
    supported = {option for action in parser._actions for option in action.option_strings}

    assert "--config" in launcher
    assert "--run-id" in launcher
    assert "--device" not in launcher
    assert {"--config", "--run-id"} <= supported


def test_physical_gpu_mapping_keeps_configured_logical_cuda_zero() -> None:
    """CUDA_VISIBLE_DEVICES selects the physical GPU; YAML uses logical cuda:0."""

    launcher = LAUNCHER.read_text(encoding="utf-8")
    config = load_config(B0_CONFIG)

    assert 'export CUDA_VISIBLE_DEVICES="${physical_gpu}"' in launcher
    assert config.trainer.device == "cuda:0"
