from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
from config_tools import ConfigError, config_digest, load_config


def cross_field_errors(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    model = config["model"]
    noise = config["noise"]
    loss = config["loss"]

    if model["backbone"]["name"] != "ViT-B/32" or model["backbone"]["pretrained"] != "openai":
        errors.append("only official OpenAI CLIP ViT-B/32 is allowed")

    lora = model["lora"]
    if lora["enabled"]:
        if lora["rank"] <= 0 or lora["alpha"] <= 0:
            errors.append("enabled LoRA requires positive rank and alpha")
        if not lora["target_blocks"] or not lora["target_projections"]:
            errors.append("enabled LoRA requires explicit target blocks and projections")
    elif lora["rank"] != 0 or lora["target_blocks"]:
        errors.append("disabled LoRA must not retain active rank or target blocks")

    if model["teacher"].get("inference_export", False):
        errors.append("training teacher must never be exported into final inference")

    enabled_signal_coefficients = [
        float(item.get("coefficient", 0.0))
        for item in noise["signals"].values()
        if item.get("enabled", False)
    ]
    if noise["enabled"] and sum(enabled_signal_coefficients) <= 0:
        errors.append("noise modeling is enabled but no positive signal coefficient exists")

    pseudo = noise["pseudolabel"]
    if pseudo["enabled"]:
        if not 0 < pseudo["maximum_dataset_fraction"] <= 0.25:
            errors.append("pseudo-label dataset fraction must be in (0, 0.25]")
        if not 0 < pseudo["pseudo_mix"] < 1:
            errors.append("pseudo_mix must preserve both original and pseudo targets")

    if loss["feature_anchor"]["enabled"] and not model["teacher"]["enabled"]:
        errors.append("feature anchor requires an enabled frozen teacher")
    if loss["consistency"]["enabled"] and not config["data"]["strong_transform"]["enabled"]:
        errors.append("consistency loss requires strong_transform")
    if config["evaluation"].get("test_time_augmentation", False):
        errors.append("TTA is disabled in the conservative compliant baseline")

    paths = config["paths"]
    resolved = [paths.get(name) for name in ("train_root", "test_root", "run_root")]
    plain = [value for value in resolved if isinstance(value, str) and not value.startswith("${")]
    if len(set(plain)) != len(plain):
        errors.append("train_root, test_root, and run_root must be distinct")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--require-env", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    schema_path = (
        Path(args.schema).resolve()
        if args.schema
        else (script_dir.parent / "configs" / "schema" / "experiment.schema.json")
    )
    try:
        config = load_config(args.config, strict_env=args.require_env)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(config)
        errors = cross_field_errors(config)
        if errors:
            raise ConfigError("; ".join(errors))
    except (ConfigError, jsonschema.ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"CONFIG_INVALID: {exc}")
        return 2

    print(json.dumps({"status": "ok", "digest": config_digest(config)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
