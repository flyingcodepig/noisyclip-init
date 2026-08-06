# NoisyCLIP

NoisyCLIP is the project skeleton for the F01 milestone of a noisy-label
fine-grained image classification pipeline based on official OpenAI CLIP
ViT-B/32 weights.

This repository currently contains only stable public interfaces, strict
configuration loading, and command-line entry skeletons. It intentionally does
not implement training, inference, export, data auditing, submission
validation, CLIP weight download, or competition-data access.

## Development

Install editable development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the F01 quality gates:

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
```

## CLI Skeleton

The public command modules exist and support `--help`:

```bash
python -m noisyclip.cli.audit_data --help
python -m noisyclip.cli.train --help
python -m noisyclip.cli.evaluate --help
python -m noisyclip.cli.infer --help
python -m noisyclip.cli.export --help
python -m noisyclip.cli.validate_submission --help
```

Running the commands for real currently raises a clear `NotImplementedError` or
returns only skeleton status. Later F02 agents should keep these module names
stable and add implementation behind the existing interfaces.
